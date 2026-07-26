//! WASI preview1 function implementations for sf-nano-core.
//!
//! Each public function has the signature `HostFn`:
//! `fn(&mut Caller, &[Value], &mut [Value]) -> Result<(), WasmError>`

use crate::collections;

use filetime::{set_file_handle_times, set_file_times, set_symlink_file_times, FileTime};
use std::format;
use std::hash::{Hash, Hasher};
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::vec::Vec;

use crate::error::WasmError;
use crate::vm::entities::Caller;
use crate::vm::value::Value;

use super::FdEntry;

// ---------------------------------------------------------------------------
// WASI errno constants
// ---------------------------------------------------------------------------

const ERRNO_ACCES: i32 = 2;
const ERRNO_SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8;
const ERRNO_EXIST: i32 = 20;
const ERRNO_ILSEQ: i32 = 25;
const ERRNO_INVAL: i32 = 28;
const ERRNO_IO: i32 = 29;
const ERRNO_ISDIR: i32 = 31;
const ERRNO_LOOP: i32 = 32;
const ERRNO_NAMETOOLONG: i32 = 37;
const ERRNO_NOENT: i32 = 44;
const ERRNO_NOTDIR: i32 = 54;
const ERRNO_NOTEMPTY: i32 = 55;
const ERRNO_NOTSOCK: i32 = 57;
const ERRNO_PERM: i32 = 63;
const ERRNO_SPIPE: i32 = 70;
const ERRNO_NOTCAPABLE: i32 = 76;

// ---------------------------------------------------------------------------
// WASI rights constants
// ---------------------------------------------------------------------------

const RIGHT_FD_DATASYNC: u64 = 1 << 0;
const RIGHT_FD_READ: u64 = 1 << 1;
const RIGHT_FD_SEEK: u64 = 1 << 2;
const RIGHT_FD_FDSTAT_SET_FLAGS: u64 = 1 << 3;
const RIGHT_FD_SYNC: u64 = 1 << 4;
const RIGHT_FD_TELL: u64 = 1 << 5;
const RIGHT_FD_WRITE: u64 = 1 << 6;
const RIGHT_FD_ADVISE: u64 = 1 << 7;
const RIGHT_FD_ALLOCATE: u64 = 1 << 8;
const RIGHT_PATH_CREATE_DIRECTORY: u64 = 1 << 9;
const RIGHT_PATH_CREATE_FILE: u64 = 1 << 10;
const RIGHT_PATH_LINK_SOURCE: u64 = 1 << 11;
const RIGHT_PATH_LINK_TARGET: u64 = 1 << 12;
const RIGHT_PATH_OPEN: u64 = 1 << 13;
const RIGHT_FD_READDIR: u64 = 1 << 14;
const RIGHT_PATH_READLINK: u64 = 1 << 15;
const RIGHT_PATH_RENAME_SOURCE: u64 = 1 << 16;
const RIGHT_PATH_RENAME_TARGET: u64 = 1 << 17;
const RIGHT_PATH_FILESTAT_GET: u64 = 1 << 18;
const RIGHT_PATH_FILESTAT_SET_SIZE: u64 = 1 << 19;
const RIGHT_PATH_FILESTAT_SET_TIMES: u64 = 1 << 20;
const RIGHT_FD_FILESTAT_GET: u64 = 1 << 21;
const RIGHT_FD_FILESTAT_SET_SIZE: u64 = 1 << 22;
const RIGHT_FD_FILESTAT_SET_TIMES: u64 = 1 << 23;
const RIGHT_PATH_SYMLINK: u64 = 1 << 24;
const RIGHT_PATH_REMOVE_DIRECTORY: u64 = 1 << 25;
const RIGHT_PATH_UNLINK_FILE: u64 = 1 << 26;
const RIGHT_POLL_FD_READWRITE: u64 = 1 << 27;

/// All rights applicable to a regular file.
const RIGHTS_FILE_BASE: u64 = RIGHT_FD_DATASYNC
    | RIGHT_FD_READ
    | RIGHT_FD_SEEK
    | RIGHT_FD_FDSTAT_SET_FLAGS
    | RIGHT_FD_SYNC
    | RIGHT_FD_TELL
    | RIGHT_FD_WRITE
    | RIGHT_FD_ADVISE
    | RIGHT_FD_ALLOCATE
    | RIGHT_FD_FILESTAT_GET
    | RIGHT_FD_FILESTAT_SET_SIZE
    | RIGHT_FD_FILESTAT_SET_TIMES
    | RIGHT_POLL_FD_READWRITE;

/// All rights applicable to a directory (for preopen inheriting).
const RIGHTS_DIR_BASE: u64 = RIGHT_PATH_CREATE_DIRECTORY
    | RIGHT_PATH_CREATE_FILE
    | RIGHT_PATH_LINK_SOURCE
    | RIGHT_PATH_LINK_TARGET
    | RIGHT_PATH_OPEN
    | RIGHT_FD_READDIR
    | RIGHT_PATH_READLINK
    | RIGHT_PATH_RENAME_SOURCE
    | RIGHT_PATH_RENAME_TARGET
    | RIGHT_PATH_FILESTAT_GET
    | RIGHT_PATH_FILESTAT_SET_SIZE
    | RIGHT_PATH_FILESTAT_SET_TIMES
    | RIGHT_PATH_SYMLINK
    | RIGHT_PATH_REMOVE_DIRECTORY
    | RIGHT_PATH_UNLINK_FILE
    | RIGHT_FD_FDSTAT_SET_FLAGS
    | RIGHT_FD_SYNC
    | RIGHT_FD_DATASYNC
    | RIGHT_FD_FILESTAT_GET
    | RIGHT_FD_FILESTAT_SET_TIMES;

/// Rights a preopen directory inherits to files/dirs opened under it.
const RIGHTS_DIR_INHERITING: u64 = RIGHTS_DIR_BASE | RIGHTS_FILE_BASE;

// ---------------------------------------------------------------------------
// WASI oflags / fdflags constants
// ---------------------------------------------------------------------------

const OFLAGS_CREAT: i32 = 1;
const OFLAGS_DIRECTORY: i32 = 2;
const OFLAGS_EXCL: i32 = 4;
const OFLAGS_TRUNC: i32 = 8;

const FDFLAGS_APPEND: u16 = 1;
const FDFLAGS_DSYNC: u16 = 2;
const FDFLAGS_NONBLOCK: u16 = 4;
const FDFLAGS_RSYNC: u16 = 8;
const FDFLAGS_SYNC: u16 = 16;

// WASI filetype constants
const FILETYPE_UNKNOWN: u8 = 0;
const FILETYPE_CHARACTER_DEVICE: u8 = 2;
const FILETYPE_DIRECTORY: u8 = 3;
const FILETYPE_REGULAR_FILE: u8 = 4;
const FILETYPE_SYMBOLIC_LINK: u8 = 7;

const PATH_RIGHTS: u64 = RIGHT_PATH_CREATE_DIRECTORY
    | RIGHT_PATH_CREATE_FILE
    | RIGHT_PATH_LINK_SOURCE
    | RIGHT_PATH_LINK_TARGET
    | RIGHT_PATH_OPEN
    | RIGHT_PATH_READLINK
    | RIGHT_PATH_RENAME_SOURCE
    | RIGHT_PATH_RENAME_TARGET
    | RIGHT_PATH_FILESTAT_GET
    | RIGHT_PATH_FILESTAT_SET_SIZE
    | RIGHT_PATH_FILESTAT_SET_TIMES
    | RIGHT_PATH_SYMLINK
    | RIGHT_PATH_REMOVE_DIRECTORY
    | RIGHT_PATH_UNLINK_FILE;

// ---------------------------------------------------------------------------
// Memory helper functions
// ---------------------------------------------------------------------------

fn as_i32(v: &Value) -> Result<i32, WasmError> {
    match v {
        Value::I32(n) => Ok(*n),
        _ => Err(WasmError::trap("expected i32 argument")),
    }
}

fn as_i64(v: &Value) -> Result<i64, WasmError> {
    match v {
        Value::I64(n) => Ok(*n),
        _ => Err(WasmError::trap("expected i64 argument")),
    }
}

fn read_mem(mem: &[u8], ptr: u32, len: u32) -> Result<&[u8], WasmError> {
    let start = ptr as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| WasmError::trap("memory access out of bounds"))?;
    if end > mem.len() {
        return Err(WasmError::trap("memory access out of bounds"));
    }
    Ok(&mem[start..end])
}

fn write_mem(mem: &mut [u8], ptr: u32, data: &[u8]) -> Result<(), WasmError> {
    let start = ptr as usize;
    let end = start
        .checked_add(data.len())
        .ok_or_else(|| WasmError::trap("memory access out of bounds"))?;
    if end > mem.len() {
        return Err(WasmError::trap("memory access out of bounds"));
    }
    mem[start..end].copy_from_slice(data);
    Ok(())
}

fn write_u32_le(mem: &mut [u8], ptr: u32, val: u32) -> Result<(), WasmError> {
    write_mem(mem, ptr, &val.to_le_bytes())
}

fn write_u64_le(mem: &mut [u8], ptr: u32, val: u64) -> Result<(), WasmError> {
    write_mem(mem, ptr, &val.to_le_bytes())
}

fn write_u16_le(mem: &mut [u8], ptr: u32, val: u16) -> Result<(), WasmError> {
    write_mem(mem, ptr, &val.to_le_bytes())
}

fn write_u8_le(mem: &mut [u8], ptr: u32, val: u8) -> Result<(), WasmError> {
    write_mem(mem, ptr, &[val])
}

fn read_u32_le(mem: &[u8], ptr: u32) -> Result<u32, WasmError> {
    let slice = read_mem(mem, ptr, 4)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn get_mem<'a>(caller: &'a mut Caller) -> Result<&'a mut [u8], WasmError> {
    caller
        .memory_mut()
        .ok_or_else(|| WasmError::trap("no linear memory available"))
}

// ---------------------------------------------------------------------------
// Path resolution helper
// ---------------------------------------------------------------------------

/// Resolve a relative guest path under a base host directory.
/// Rejects absolute paths and `..` components that escape the base.
fn resolve_under_base(base: &Path, rel: &str) -> Result<PathBuf, i32> {
    let is_abs = {
        let p = Path::new(rel);
        if p.is_absolute() {
            true
        } else {
            rel.starts_with('/') || rel.starts_with('\\')
        }
    };
    if is_abs {
        return Err(ERRNO_NOTCAPABLE);
    }
    let mut parts = collections::Vec::new();
    for comp in rel.split(['/', '\\']) {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp == ".." {
            if parts.is_empty() {
                return Err(ERRNO_NOTCAPABLE);
            }
            parts.pop();
        } else {
            parts.push(comp);
        }
    }
    let mut out = PathBuf::from(base);
    for c in parts {
        out.push(c);
    }
    Ok(out)
}

fn path_error_to_errno(e: &std::io::Error) -> i32 {
    use std::io::ErrorKind as K;

    match e.kind() {
        K::AlreadyExists => ERRNO_EXIST,
        K::DirectoryNotEmpty => ERRNO_NOTEMPTY,
        K::InvalidInput => ERRNO_INVAL,
        K::IsADirectory => ERRNO_ISDIR,
        K::NotADirectory => ERRNO_NOTDIR,
        K::NotFound => ERRNO_NOENT,
        K::PermissionDenied => ERRNO_PERM,
        _ => ERRNO_IO,
    }
}

fn validate_path_bytes(bytes: &[u8]) -> Result<String, i32> {
    if bytes.contains(&0) {
        return Err(ERRNO_INVAL);
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Err(ERRNO_ILSEQ),
    }
}

#[inline]
fn ns_to_filetime(ns: u64) -> FileTime {
    let secs = (ns / 1_000_000_000) as i64;
    let nsec = (ns % 1_000_000_000) as u32;
    FileTime::from_unix_time(secs, nsec)
}

fn realtime_ns() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() * 1_000_000_000 + d.subsec_nanos() as u64,
        Err(_) => 0,
    }
}

fn monotonic_ns() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    static LAST: OnceLock<Mutex<u64>> = OnceLock::new();

    let start = START.get_or_init(Instant::now);
    let elapsed = start.elapsed().as_nanos() as u64;
    let mut last = LAST.get_or_init(|| Mutex::new(0)).lock().unwrap();
    let next = if elapsed > *last {
        elapsed
    } else {
        (*last).saturating_add(1)
    };
    *last = next;
    next
}

#[cfg(not(all(target_os = "linux", target_arch = "riscv32")))]
fn timestamp_ns(time: std::io::Result<SystemTime>) -> u64 {
    time.ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn write_filestat(
    mem: &mut [u8],
    buf_ptr: u32,
    dev: u64,
    ino: u64,
    filetype: u8,
    nlink: u64,
    size: u64,
    atim: u64,
    mtim: u64,
    ctim: u64,
) -> Result<(), WasmError> {
    write_u64_le(mem, buf_ptr, dev)?;
    write_u64_le(mem, buf_ptr + 8, ino)?;
    write_u8_le(mem, buf_ptr + 16, filetype)?;
    write_u64_le(mem, buf_ptr + 24, nlink)?;
    write_u64_le(mem, buf_ptr + 32, size)?;
    write_u64_le(mem, buf_ptr + 40, atim)?;
    write_u64_le(mem, buf_ptr + 48, mtim)?;
    write_u64_le(mem, buf_ptr + 56, ctim)?;
    Ok(())
}

fn preopen_index_for_fd(ctx: &super::WasiCtx, fd: i32) -> Option<usize> {
    let idx = fd - 3;
    if idx < 0 || ctx.closed_preopens.contains(&fd) {
        return None;
    }
    let idx = idx as usize;
    (idx < ctx.preopens.len()).then_some(idx)
}

fn dir_fd_state(fd: i32) -> Result<(PathBuf, u64, u64), i32> {
    super::with_ctx(|ctx| {
        if let Some(idx) = preopen_index_for_fd(ctx, fd) {
            return Ok((
                ctx.preopens[idx].host_path.clone(),
                RIGHTS_DIR_BASE,
                RIGHTS_DIR_INHERITING,
            ));
        }
        match ctx.fds.get(&fd) {
            Some(FdEntry::Dir {
                host_path,
                rights_base,
                rights_inh,
            }) => Ok((host_path.clone(), *rights_base, *rights_inh)),
            Some(FdEntry::File { .. }) => Err(ERRNO_NOTDIR),
            None => Err(ERRNO_BADF),
        }
    })
}

#[derive(Clone, Copy)]
struct HostPathKind {
    is_file: bool,
    is_dir: bool,
    is_symlink: bool,
}

impl HostPathKind {
    fn filetype(self) -> u8 {
        if self.is_symlink {
            FILETYPE_SYMBOLIC_LINK
        } else if self.is_dir {
            FILETYPE_DIRECTORY
        } else if self.is_file {
            FILETYPE_REGULAR_FILE
        } else {
            FILETYPE_UNKNOWN
        }
    }
}

#[derive(Clone, Copy)]
struct HostStat {
    kind: HostPathKind,
    ino: u64,
    size: u64,
    atim: u64,
    mtim: u64,
    ctim: u64,
}

fn write_host_filestat(mem: &mut [u8], buf_ptr: u32, stat: HostStat) -> Result<(), WasmError> {
    write_filestat(
        mem,
        buf_ptr,
        1,
        stat.ino,
        stat.kind.filetype(),
        1,
        stat.size,
        stat.atim,
        stat.mtim,
        stat.ctim,
    )
}

#[cfg(all(target_os = "linux", target_arch = "riscv32"))]
mod rv32_linux_stat {
    // RV32 Linux/musl under qemu-riscv32-static has a narrow host-side
    // incompatibility in Rust's std metadata path. The real trace showed the
    // guest process issuing `statx(...) = 0` and then faulting or corrupting
    // runtime-call state while `std::fs::{metadata, symlink_metadata}` /
    // `File::metadata` decoded the result. WASI needs exactly this information
    // for path classification and filestat, so RV32/Linux uses the kernel
    // `statx` ABI directly here. Keep this target-specific; all other targets
    // stay on `std::fs`, and any removal must be revalidated with
    // `sf-nano-wasitest` under qemu-riscv32-static.
    use std::ffi::CString;
    use std::mem::MaybeUninit;
    use std::os::fd::AsRawFd;
    use std::os::raw::{c_char, c_long};
    use std::os::unix::ffi::OsStrExt;

    #[repr(C)]
    pub(super) struct StatxTimestamp {
        tv_sec: i64,
        tv_nsec: u32,
        __pad: i32,
    }

    #[repr(C)]
    pub(super) struct Statx {
        stx_mask: u32,
        stx_blksize: u32,
        stx_attributes: u64,
        stx_nlink: u32,
        stx_uid: u32,
        stx_gid: u32,
        stx_mode: u16,
        __pad1: [u16; 1],
        stx_ino: u64,
        stx_size: u64,
        stx_blocks: u64,
        stx_attributes_mask: u64,
        stx_atime: StatxTimestamp,
        stx_btime: StatxTimestamp,
        stx_ctime: StatxTimestamp,
        stx_mtime: StatxTimestamp,
        stx_rdev_major: u32,
        stx_rdev_minor: u32,
        stx_dev_major: u32,
        stx_dev_minor: u32,
        stx_mnt_id: u64,
        stx_dio_mem_align: u32,
        stx_dio_offset_align: u32,
        __pad3: [u64; 12],
    }

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
    }

    const SYS_STATX: c_long = 291;
    const AT_FDCWD: c_long = -100;
    const AT_EMPTY_PATH: c_long = 0x1000;
    const AT_SYMLINK_NOFOLLOW: c_long = 0x100;
    const AT_NO_AUTOMOUNT: c_long = 0x800;
    const STATX_BASIC_STATS: c_long = 0x07ff;
    const S_IFMT: u16 = 0o170000;
    const S_IFDIR: u16 = 0o040000;
    const S_IFREG: u16 = 0o100000;
    const S_IFLNK: u16 = 0o120000;

    fn timestamp_ns(ts: &StatxTimestamp) -> u64 {
        if ts.tv_sec < 0 {
            0
        } else {
            (ts.tv_sec as u64)
                .saturating_mul(1_000_000_000)
                .saturating_add(u64::from(ts.tv_nsec))
        }
    }

    fn stat_from_statx(statx: Statx) -> super::HostStat {
        let mode = statx.stx_mode & S_IFMT;
        let kind = super::HostPathKind {
            is_file: mode == S_IFREG,
            is_dir: mode == S_IFDIR,
            is_symlink: mode == S_IFLNK,
        };
        super::HostStat {
            kind,
            ino: statx.stx_ino,
            size: if kind.is_file { statx.stx_size } else { 0 },
            atim: timestamp_ns(&statx.stx_atime),
            mtim: timestamp_ns(&statx.stx_mtime),
            ctim: timestamp_ns(&statx.stx_ctime),
        }
    }

    fn statx_raw(
        fd: c_long,
        path: *const c_char,
        flags: c_long,
    ) -> Result<super::HostStat, std::io::Error> {
        let mut statx_buf = MaybeUninit::<Statx>::zeroed();
        let rc = unsafe {
            syscall(
                SYS_STATX,
                fd,
                path,
                flags,
                STATX_BASIC_STATS,
                statx_buf.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(stat_from_statx(unsafe { statx_buf.assume_init() }))
    }

    pub(super) fn path(
        host_path: &std::path::Path,
        follow_symlink: bool,
    ) -> Result<super::HostStat, std::io::Error> {
        let c_path = CString::new(host_path.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        let flags = AT_NO_AUTOMOUNT
            | if follow_symlink {
                0
            } else {
                AT_SYMLINK_NOFOLLOW
            };
        statx_raw(AT_FDCWD, c_path.as_ptr(), flags)
    }

    pub(super) fn file(file: &std::fs::File) -> Result<super::HostStat, std::io::Error> {
        let empty = b"\0";
        statx_raw(
            file.as_raw_fd() as c_long,
            empty.as_ptr().cast::<c_char>(),
            AT_EMPTY_PATH | AT_NO_AUTOMOUNT,
        )
    }
}

#[cfg(all(target_os = "linux", target_arch = "riscv32"))]
fn stat_path_metadata(host_path: &Path, follow_symlink: bool) -> Result<HostStat, std::io::Error> {
    rv32_linux_stat::path(host_path, follow_symlink)
}

#[cfg(all(target_os = "linux", target_arch = "riscv32"))]
fn stat_file_metadata(file: &std::fs::File, _host_path: &Path) -> Result<HostStat, std::io::Error> {
    rv32_linux_stat::file(file)
}

#[cfg(not(all(target_os = "linux", target_arch = "riscv32")))]
fn stat_path_metadata(host_path: &Path, follow_symlink: bool) -> Result<HostStat, std::io::Error> {
    let meta = if follow_symlink {
        std::fs::metadata(host_path)?
    } else {
        std::fs::symlink_metadata(host_path)?
    };
    Ok(host_stat_from_metadata(&meta, host_path))
}

#[cfg(not(all(target_os = "linux", target_arch = "riscv32")))]
fn stat_file_metadata(file: &std::fs::File, host_path: &Path) -> Result<HostStat, std::io::Error> {
    let meta = file.metadata()?;
    Ok(host_stat_from_metadata(&meta, host_path))
}

fn stat_path_kind(host_path: &Path, follow_symlink: bool) -> Result<HostPathKind, std::io::Error> {
    Ok(stat_path_metadata(host_path, follow_symlink)?.kind)
}

#[cfg(windows)]
mod windows_file_id {
    // Windows has no inode, but it does have a per-volume file index that two
    // names for one file share -- which is exactly the identity `path_link`
    // requires and a path-derived number cannot express. `std::os::windows`
    // exposes it only behind the unstable `windows_by_handle` feature, so ask
    // the OS directly, the way `linux_link` below does for `linkat`.
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::Path;

    // Needed to open a *directory* handle at all; harmless for files.
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    // Identify the link itself rather than what it points at.
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

    #[repr(C)]
    #[derive(Default)]
    struct Filetime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ByHandleFileInformation {
        file_attributes: u32,
        creation_time: Filetime,
        last_access_time: Filetime,
        last_write_time: Filetime,
        volume_serial_number: u32,
        file_size_high: u32,
        file_size_low: u32,
        number_of_links: u32,
        file_index_high: u32,
        file_index_low: u32,
    }

    unsafe extern "system" {
        fn GetFileInformationByHandle(
            handle: *mut core::ffi::c_void,
            info: *mut ByHandleFileInformation,
        ) -> i32;
    }

    /// The volume-relative file index, or `None` if the file cannot be opened
    /// or queried. Callers fall back to a synthesized number, so a failure
    /// here costs hard-link identity rather than the whole stat.
    pub(super) fn file_index(path: &Path, no_follow: bool) -> Option<u64> {
        let mut flags = FILE_FLAG_BACKUP_SEMANTICS;
        if no_follow {
            flags |= FILE_FLAG_OPEN_REPARSE_POINT;
        }
        // `access_mode(0)` asks for no data access at all; the handle can
        // still answer metadata, and it avoids failing on a file the guest
        // is not permitted to read.
        let file = std::fs::OpenOptions::new()
            .access_mode(0)
            .custom_flags(flags)
            .open(path)
            .ok()?;

        let mut info = ByHandleFileInformation::default();
        // SAFETY: `file` owns a live handle for the duration of the call, and
        // `info` is a correctly-shaped, fully-initialized output buffer.
        let ok =
            unsafe { GetFileInformationByHandle(file.as_raw_handle().cast(), &mut info as *mut _) };
        if ok == 0 {
            return None;
        }
        Some(((info.file_index_high as u64) << 32) | info.file_index_low as u64)
    }
}

#[cfg(target_os = "linux")]
mod linux_link {
    // WASI `path_link` does not follow the source path in this implementation.
    // Use `linkat` directly so Linux receives that contract explicitly instead
    // of depending on `std::fs::hard_link` behavior.
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    use std::os::unix::ffi::OsStrExt;

    unsafe extern "C" {
        fn linkat(
            olddirfd: c_int,
            oldpath: *const c_char,
            newdirfd: c_int,
            newpath: *const c_char,
            flags: c_int,
        ) -> c_int;
    }

    const AT_FDCWD: c_int = -100;

    pub(super) fn hard_link_no_follow(
        old_path: &std::path::Path,
        new_path: &std::path::Path,
    ) -> Result<(), std::io::Error> {
        let old_c = CString::new(old_path.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        let new_c = CString::new(new_path.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        let rc = unsafe { linkat(AT_FDCWD, old_c.as_ptr(), AT_FDCWD, new_c.as_ptr(), 0) };
        if rc == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

#[cfg(target_os = "linux")]
fn hard_link_no_follow(old_path: &Path, new_path: &Path) -> Result<(), std::io::Error> {
    linux_link::hard_link_no_follow(old_path, new_path)
}

#[cfg(not(target_os = "linux"))]
fn hard_link_no_follow(old_path: &Path, new_path: &Path) -> Result<(), std::io::Error> {
    std::fs::hard_link(old_path, new_path)
}

#[cfg(not(all(target_os = "linux", target_arch = "riscv32")))]
fn host_stat_from_metadata(meta: &std::fs::Metadata, host_path: &Path) -> HostStat {
    let kind = HostPathKind {
        is_file: meta.is_file(),
        is_dir: meta.is_dir(),
        is_symlink: meta.file_type().is_symlink(),
    };
    HostStat {
        kind,
        ino: derive_ino_from_meta(meta, host_path),
        size: if kind.is_file { meta.len() } else { 0 },
        atim: timestamp_ns(meta.accessed()),
        mtim: timestamp_ns(meta.modified()),
        ctim: timestamp_ns(meta.created()),
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "riscv32")))]
fn derive_ino_from_meta(meta: &std::fs::Metadata, _host_path: &Path) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.ino()
    }
    #[cfg(not(unix))]
    {
        // Ask the OS for real file identity where it has one. Hashing the
        // path below cannot answer it: a hard link is two names for one
        // file, so a path-derived number reports them as different files.
        #[cfg(windows)]
        if let Some(index) = windows_file_id::file_index(_host_path, meta.file_type().is_symlink())
        {
            return index & ((1u64 << 53) - 1);
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        _host_path.to_string_lossy().hash(&mut hasher);
        let ft: u8 = if meta.is_dir() {
            FILETYPE_DIRECTORY
        } else if meta.is_file() {
            FILETYPE_REGULAR_FILE
        } else {
            FILETYPE_UNKNOWN
        };
        ft.hash(&mut hasher);
        meta.len().hash(&mut hasher);
        timestamp_ns(meta.created()).hash(&mut hasher);
        timestamp_ns(meta.modified()).hash(&mut hasher);
        hasher.finish() & ((1u64 << 53) - 1)
    }
}

fn derive_ino_for_path(host_path: &Path) -> u64 {
    match stat_path_metadata(host_path, false) {
        Ok(stat) => stat.ino,
        Err(_) => {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            host_path.to_string_lossy().hash(&mut hasher);
            hasher.finish() & ((1u64 << 53) - 1)
        }
    }
}

// ===========================================================================
// WASI preview1 functions — fully implemented
// ===========================================================================

// ---------------------------------------------------------------------------
// proc_exit
// ---------------------------------------------------------------------------

pub(crate) fn proc_exit(
    _caller: &mut Caller,
    args: &[Value],
    _results: &mut [Value],
) -> Result<(), WasmError> {
    let code = as_i32(&args[0])?;
    Err(WasmError::exit_with_code(code))
}

// ---------------------------------------------------------------------------
// args_sizes_get
// ---------------------------------------------------------------------------

pub(crate) fn args_sizes_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let argc_ptr = as_i32(&args[0])? as u32;
    let buf_size_ptr = as_i32(&args[1])? as u32;

    let (argc, buf_size) = super::with_ctx(|ctx| {
        let argc = ctx.args.len() as u32;
        let buf_size: u32 = ctx.args.iter().map(|a| a.len() as u32 + 1).sum();
        (argc, buf_size)
    });

    let mem = get_mem(caller)?;
    write_u32_le(mem, argc_ptr, argc)?;
    write_u32_le(mem, buf_size_ptr, buf_size)?;

    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// args_get
// ---------------------------------------------------------------------------

pub(crate) fn args_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let argv_ptr = as_i32(&args[0])? as u32;
    let buf_ptr = as_i32(&args[1])? as u32;

    let argv: collections::Vec<String> = super::with_ctx(|ctx| ctx.args.clone());

    let mem = get_mem(caller)?;
    let mut buf_offset = buf_ptr;
    for (i, arg) in argv.iter().enumerate() {
        // write pointer to this arg's string data
        write_u32_le(mem, argv_ptr + (i as u32) * 4, buf_offset)?;
        // write the string data + NUL
        let bytes = arg.as_bytes();
        write_mem(mem, buf_offset, bytes)?;
        write_u8_le(mem, buf_offset + bytes.len() as u32, 0)?;
        buf_offset += bytes.len() as u32 + 1;
    }

    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// environ_sizes_get
// ---------------------------------------------------------------------------

pub(crate) fn environ_sizes_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let count_ptr = as_i32(&args[0])? as u32;
    let size_ptr = as_i32(&args[1])? as u32;

    let (count, size) = super::with_ctx(|ctx| {
        let count = ctx.env.len() as u32;
        // each entry: "KEY=VALUE\0"
        let size: u32 = ctx
            .env
            .iter()
            .map(|(k, v)| k.len() as u32 + 1 + v.len() as u32 + 1) // key + '=' + value + '\0'
            .sum();
        (count, size)
    });

    let mem = get_mem(caller)?;
    write_u32_le(mem, count_ptr, count)?;
    write_u32_le(mem, size_ptr, size)?;

    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// environ_get
// ---------------------------------------------------------------------------

pub(crate) fn environ_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let environ_ptr = as_i32(&args[0])? as u32;
    let buf_ptr = as_i32(&args[1])? as u32;

    let env: collections::Vec<(String, String)> = super::with_ctx(|ctx| ctx.env.clone());

    let mem = get_mem(caller)?;
    let mut buf_offset = buf_ptr;
    for (i, (k, v)) in env.iter().enumerate() {
        write_u32_le(mem, environ_ptr + (i as u32) * 4, buf_offset)?;
        let entry = format!("{}={}", k, v);
        let bytes = entry.as_bytes();
        write_mem(mem, buf_offset, bytes)?;
        write_u8_le(mem, buf_offset + bytes.len() as u32, 0)?;
        buf_offset += bytes.len() as u32 + 1;
    }

    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_write
// ---------------------------------------------------------------------------

pub(crate) fn fd_write(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let iovs_ptr = as_i32(&args[1])? as u32;
    let iovs_len = as_i32(&args[2])? as u32;
    let nwritten_ptr = as_i32(&args[3])? as u32;

    let mem = get_mem(caller)?;

    // Gather iov entries
    let mut total_written: u32 = 0;

    match fd {
        1 | 2 => {
            // stdout / stderr
            let is_closed = super::with_ctx(|ctx| ctx.closed_stdio.contains(&fd));
            if is_closed {
                results[0] = Value::I32(ERRNO_BADF);
                return Ok(());
            }
            let mut out_buf = collections::Vec::new();
            for i in 0..iovs_len {
                let base = iovs_ptr + i * 8;
                let ptr = read_u32_le(mem, base)?;
                let len = read_u32_le(mem, base + 4)?;
                let data = read_mem(mem, ptr, len)?;
                out_buf.extend_from_slice(data);
                total_written += len;
            }
            if fd == 1 {
                let _ = std::io::stdout().write_all(&out_buf);
                let _ = std::io::stdout().flush();
            } else {
                let _ = std::io::stderr().write_all(&out_buf);
                let _ = std::io::stderr().flush();
            }
        }
        _ => {
            // Collect data first, then access ctx
            let mut buffers = collections::Vec::new();
            for i in 0..iovs_len {
                let base = iovs_ptr + i * 8;
                let ptr = read_u32_le(mem, base)?;
                let len = read_u32_le(mem, base + 4)?;
                let data = read_mem(mem, ptr, len)?.to_vec();
                total_written += len;
                buffers.push(data);
            }

            let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
                Some(FdEntry::File {
                    file,
                    rights_base,
                    fdflags,
                    ..
                }) => {
                    if (*rights_base & RIGHT_FD_WRITE) == 0 {
                        return ERRNO_NOTCAPABLE;
                    }
                    if (*fdflags & FDFLAGS_APPEND) != 0 {
                        let _ = file.seek(SeekFrom::End(0));
                    }
                    for buf in &buffers {
                        if file.write_all(buf).is_err() {
                            return ERRNO_IO;
                        }
                    }
                    ERRNO_SUCCESS
                }
                Some(FdEntry::Dir { .. }) => ERRNO_ISDIR,
                None => ERRNO_BADF,
            });

            if errno != ERRNO_SUCCESS {
                results[0] = Value::I32(errno);
                return Ok(());
            }
        }
    }

    write_u32_le(mem, nwritten_ptr, total_written)?;
    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_read
// ---------------------------------------------------------------------------

pub(crate) fn fd_read(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let iovs_ptr = as_i32(&args[1])? as u32;
    let iovs_len = as_i32(&args[2])? as u32;
    let nread_ptr = as_i32(&args[3])? as u32;

    let mem = get_mem(caller)?;

    match fd {
        0 => {
            // stdin: read from real stdin
            let is_closed = super::with_ctx(|ctx| ctx.closed_stdio.contains(&0));
            if is_closed {
                results[0] = Value::I32(ERRNO_BADF);
                return Ok(());
            }
            let mut iovs = collections::Vec::new();
            for i in 0..iovs_len {
                let base = iovs_ptr + i * 8;
                let ptr = read_u32_le(mem, base)?;
                let len = read_u32_le(mem, base + 4)?;
                iovs.push((ptr, len));
            }
            // Use buffered stdin to provide consistent read behavior.
            // We read all remaining stdin data into a static buffer on first call,
            // then serve subsequent reads from the buffer.
            use std::sync::Mutex;
            static STDIN_BUF: Mutex<Option<(Vec<u8>, usize)>> = Mutex::new(None);

            let mut total_read: u32 = 0;
            let mut guard = STDIN_BUF.lock().unwrap();
            let (buf, pos) = guard.get_or_insert_with(|| {
                let mut data = Vec::new();
                std::io::stdin().read_to_end(&mut data).unwrap_or_default();
                (data, 0)
            });

            for &(ptr, len) in &iovs {
                if *pos >= buf.len() {
                    break; // EOF
                }
                let start = ptr as usize;
                let end = start + len as usize;
                if end > mem.len() {
                    results[0] = Value::I32(ERRNO_INVAL);
                    return Ok(());
                }
                let avail = std::cmp::min(len as usize, buf.len() - *pos);
                mem[start..start + avail].copy_from_slice(&buf[*pos..*pos + avail]);
                *pos += avail;
                total_read += avail as u32;
                if avail < len as usize {
                    break; // short read
                }
            }
            drop(guard);
            write_u32_le(mem, nread_ptr, total_read)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        _ => {
            // Collect iov specs
            let mut iovs = collections::Vec::new();
            for i in 0..iovs_len {
                let base = iovs_ptr + i * 8;
                let ptr = read_u32_le(mem, base)?;
                let len = read_u32_le(mem, base + 4)?;
                iovs.push((ptr, len));
            }

            let mut total_read: u32 = 0;
            let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
                Some(FdEntry::File {
                    file, rights_base, ..
                }) => {
                    if (*rights_base & RIGHT_FD_READ) == 0 {
                        return ERRNO_NOTCAPABLE;
                    }
                    for &(ptr, len) in &iovs {
                        let start = ptr as usize;
                        let end = start + len as usize;
                        if end > mem.len() {
                            return ERRNO_INVAL;
                        }
                        match file.read(&mut mem[start..end]) {
                            Ok(0) => break,
                            Ok(n) => {
                                total_read += n as u32;
                                if (n as u32) < len {
                                    break;
                                }
                            }
                            Err(_) => return ERRNO_IO,
                        }
                    }
                    ERRNO_SUCCESS
                }
                Some(FdEntry::Dir { .. }) => ERRNO_ISDIR,
                None => ERRNO_BADF,
            });

            if errno != ERRNO_SUCCESS {
                results[0] = Value::I32(errno);
                return Ok(());
            }
            write_u32_le(mem, nread_ptr, total_read)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_close
// ---------------------------------------------------------------------------

pub(crate) fn fd_close(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;

    let errno = super::with_ctx_mut(|ctx| {
        // stdio
        if (0..=2).contains(&fd) {
            ctx.closed_stdio.insert(fd);
            return ERRNO_SUCCESS;
        }
        // Preopened dir. A number in the preopen range is not necessarily
        // still a preopen: `fd_renumber` can move a descriptor onto it, which
        // retires the original and installs a live entry under that number.
        // Resolving through `preopen_index_for_fd` -- as every other fd
        // operation does -- lets that case fall through to the dynamic map
        // instead of reporting the retired preopen as a bad descriptor.
        if preopen_index_for_fd(ctx, fd).is_some() {
            ctx.closed_preopens.insert(fd);
            return ERRNO_SUCCESS;
        }
        // dynamic fd
        if ctx.fds.remove(&fd).is_some() {
            ERRNO_SUCCESS
        } else {
            ERRNO_BADF
        }
    });

    results[0] = Value::I32(errno);
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_seek
// ---------------------------------------------------------------------------

pub(crate) fn fd_seek(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let offset = as_i64(&args[1])?;
    let whence = as_i32(&args[2])?;
    let newoffset_ptr = as_i32(&args[3])? as u32;

    // stdout/stderr are not seekable
    if fd == 1 || fd == 2 {
        results[0] = Value::I32(ERRNO_SPIPE);
        return Ok(());
    }

    let seek_from = match whence {
        0 => {
            // SET
            if offset < 0 {
                results[0] = Value::I32(ERRNO_INVAL);
                return Ok(());
            }
            SeekFrom::Start(offset as u64)
        }
        1 => SeekFrom::Current(offset), // CUR
        2 => SeekFrom::End(offset),     // END
        _ => {
            results[0] = Value::I32(ERRNO_INVAL);
            return Ok(());
        }
    };

    let result = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file,
            host_path,
            rights_base,
            ..
        }) => {
            if (*rights_base & RIGHT_FD_SEEK) == 0 {
                return Err(ERRNO_NOTCAPABLE);
            }
            if offset < 0 {
                if whence == 1 {
                    if let Ok(cur) = file.stream_position() {
                        if (-offset as u64) > cur {
                            return Err(ERRNO_INVAL);
                        }
                    }
                } else if whence == 2 {
                    if let Ok(stat) = stat_file_metadata(file, host_path) {
                        if (-offset as u64) > stat.size {
                            return Err(ERRNO_INVAL);
                        }
                    }
                }
            }
            match file.seek(seek_from) {
                Ok(pos) => Ok(pos),
                Err(e) => match e.kind() {
                    std::io::ErrorKind::InvalidInput => Err(ERRNO_INVAL),
                    _ => Err(ERRNO_IO),
                },
            }
        }
        Some(FdEntry::Dir { .. }) => Err(ERRNO_ISDIR),
        None => Err(ERRNO_BADF),
    });

    match result {
        Ok(pos) => {
            let mem = get_mem(caller)?;
            write_u64_le(mem, newoffset_ptr, pos)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        Err(errno) => {
            results[0] = Value::I32(errno);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_tell
// ---------------------------------------------------------------------------

pub(crate) fn fd_tell(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let offset_ptr = as_i32(&args[1])? as u32;

    let result = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_TELL) == 0 {
                return Err(ERRNO_NOTCAPABLE);
            }
            match file.seek(SeekFrom::Current(0)) {
                Ok(pos) => Ok(pos),
                Err(_) => Err(ERRNO_IO),
            }
        }
        Some(FdEntry::Dir { .. }) => Err(ERRNO_ISDIR),
        None => Err(ERRNO_BADF),
    });

    match result {
        Ok(pos) => {
            let mem = get_mem(caller)?;
            write_u64_le(mem, offset_ptr, pos)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        Err(errno) => {
            results[0] = Value::I32(errno);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_prestat_get
// ---------------------------------------------------------------------------

pub(crate) fn fd_prestat_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let buf_ptr = as_i32(&args[1])? as u32;

    let result = super::with_ctx(|ctx| {
        if ctx.closed_preopens.contains(&fd) {
            return Err(ERRNO_BADF);
        }
        let idx = fd - 3;
        if idx < 0 || idx as usize >= ctx.preopens.len() {
            return Err(ERRNO_BADF);
        }
        let name_len = ctx.preopens[idx as usize].guest_path.len() as u32;
        Ok(name_len)
    });

    match result {
        Ok(name_len) => {
            let mem = get_mem(caller)?;
            // prestat layout: u8 tag (0=dir) + 3 pad + u32 name_len
            write_u8_le(mem, buf_ptr, 0)?;
            write_u8_le(mem, buf_ptr + 1, 0)?;
            write_u8_le(mem, buf_ptr + 2, 0)?;
            write_u8_le(mem, buf_ptr + 3, 0)?;
            write_u32_le(mem, buf_ptr + 4, name_len)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        Err(errno) => {
            results[0] = Value::I32(errno);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_prestat_dir_name
// ---------------------------------------------------------------------------

pub(crate) fn fd_prestat_dir_name(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let path_ptr = as_i32(&args[1])? as u32;
    let path_len = as_i32(&args[2])? as u32;

    let result = super::with_ctx(|ctx| {
        if ctx.closed_preopens.contains(&fd) {
            return Err(ERRNO_BADF);
        }
        let idx = fd - 3;
        if idx < 0 || idx as usize >= ctx.preopens.len() {
            return Err(ERRNO_BADF);
        }
        Ok(ctx.preopens[idx as usize].guest_path.clone())
    });

    match result {
        Ok(guest_path) => {
            let bytes = guest_path.as_bytes();
            if path_len < bytes.len() as u32 {
                results[0] = Value::I32(ERRNO_NAMETOOLONG);
                return Ok(());
            }
            let mem = get_mem(caller)?;
            write_mem(mem, path_ptr, bytes)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        Err(errno) => {
            results[0] = Value::I32(errno);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fd_fdstat_get
// ---------------------------------------------------------------------------

pub(crate) fn fd_fdstat_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let buf_ptr = as_i32(&args[1])? as u32;

    // fdstat layout (24 bytes):
    // +0:  u8  filetype
    // +1:  u8  padding
    // +2:  u16 fs_flags
    // +4:  u32 padding
    // +8:  u64 rights_base
    // +16: u64 rights_inheriting

    // Determine actual filetype for stdio based on whether they're terminals
    let stdin_type = if std::io::stdin().is_terminal() {
        FILETYPE_CHARACTER_DEVICE
    } else {
        FILETYPE_UNKNOWN
    };
    let stdout_type = if std::io::stdout().is_terminal() {
        FILETYPE_CHARACTER_DEVICE
    } else {
        FILETYPE_UNKNOWN
    };
    let stderr_type = if std::io::stderr().is_terminal() {
        FILETYPE_CHARACTER_DEVICE
    } else {
        FILETYPE_UNKNOWN
    };

    let result: Result<(u8, u16, u64, u64), i32> = super::with_ctx(|ctx| {
        match fd {
            0 => {
                // stdin
                if ctx.closed_stdio.contains(&0) {
                    return Err(ERRNO_BADF);
                }
                Ok((
                    stdin_type,
                    0,
                    RIGHT_FD_READ
                        | RIGHT_FD_FDSTAT_SET_FLAGS
                        | RIGHT_FD_FILESTAT_GET
                        | RIGHT_POLL_FD_READWRITE,
                    0,
                ))
            }
            1 => {
                // stdout
                if ctx.closed_stdio.contains(&1) {
                    return Err(ERRNO_BADF);
                }
                Ok((
                    stdout_type,
                    0,
                    RIGHT_FD_WRITE
                        | RIGHT_FD_FDSTAT_SET_FLAGS
                        | RIGHT_FD_FILESTAT_GET
                        | RIGHT_POLL_FD_READWRITE,
                    0,
                ))
            }
            2 => {
                // stderr
                if ctx.closed_stdio.contains(&2) {
                    return Err(ERRNO_BADF);
                }
                Ok((stderr_type, 0, RIGHT_FD_WRITE, 0))
            }
            _ => {
                // preopen dir?
                let preopen_end = 3 + ctx.preopens.len() as i32;
                if fd >= 3 && fd < preopen_end {
                    if ctx.closed_preopens.contains(&fd) {
                        return Err(ERRNO_BADF);
                    }
                    return Ok((
                        FILETYPE_DIRECTORY,
                        0,
                        RIGHTS_DIR_BASE,
                        RIGHTS_DIR_INHERITING,
                    ));
                }
                // dynamic fd
                match ctx.fds.get(&fd) {
                    Some(FdEntry::Dir {
                        rights_base,
                        rights_inh,
                        ..
                    }) => Ok((FILETYPE_DIRECTORY, 0, *rights_base, *rights_inh)),
                    Some(FdEntry::File {
                        rights_base,
                        rights_inh,
                        fdflags,
                        ..
                    }) => Ok((FILETYPE_REGULAR_FILE, *fdflags, *rights_base, *rights_inh)),
                    None => Err(ERRNO_BADF),
                }
            }
        }
    });

    match result {
        Ok((filetype, fs_flags, rights_base, rights_inh)) => {
            let mem = get_mem(caller)?;
            // Zero the 24-byte buffer first
            write_mem(mem, buf_ptr, &[0u8; 24])?;
            write_u8_le(mem, buf_ptr, filetype)?;
            write_u16_le(mem, buf_ptr + 2, fs_flags)?;
            write_u64_le(mem, buf_ptr + 8, rights_base)?;
            write_u64_le(mem, buf_ptr + 16, rights_inh)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        Err(errno) => {
            results[0] = Value::I32(errno);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// clock_time_get
// ---------------------------------------------------------------------------

pub(crate) fn clock_time_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let clock_id = as_i32(&args[0])?;
    let _precision = as_i64(&args[1])?;
    let time_ptr = as_i32(&args[2])? as u32;

    let nanos = if clock_id == 0 {
        realtime_ns()
    } else {
        monotonic_ns()
    };

    let mem = get_mem(caller)?;
    write_u64_le(mem, time_ptr, nanos)?;
    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// clock_res_get
// ---------------------------------------------------------------------------

pub(crate) fn clock_res_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let _clock_id = as_i32(&args[0])?;
    let time_ptr = as_i32(&args[1])? as u32;

    let mem = get_mem(caller)?;
    write_u64_le(mem, time_ptr, 1_000_000)?; // 1ms resolution
    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// random_get — simple xorshift PRNG (no external deps)
// ---------------------------------------------------------------------------

pub(crate) fn random_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let buf_ptr = as_i32(&args[0])? as u32;
    let buf_len = as_i32(&args[1])? as u32;

    let mem = get_mem(caller)?;

    // Seed from current time
    let seed = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as u64,
        Err(_) => 0xDEAD_BEEF_CAFE_BABE,
    };
    let mut state = if seed == 0 {
        0xDEAD_BEEF_CAFE_BABE
    } else {
        seed
    };

    let start = buf_ptr as usize;
    let end = start + buf_len as usize;
    if end > mem.len() {
        return Err(WasmError::trap("memory access out of bounds"));
    }

    for byte in &mut mem[start..end] {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state & 0xFF) as u8;
    }

    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

// ---------------------------------------------------------------------------
// path_open
// ---------------------------------------------------------------------------

pub(crate) fn path_open(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let dirflags = as_i32(&args[1])? as u32;
    let path_ptr = as_i32(&args[2])? as u32;
    let path_len = as_i32(&args[3])? as u32;
    let oflags = as_i32(&args[4])?;
    let rights_base = as_i64(&args[5])? as u64;
    let rights_inh = as_i64(&args[6])? as u64;
    let fdflags = as_i32(&args[7])? as u16;
    let out_fd_ptr = as_i32(&args[8])? as u32;

    let mem = get_mem(caller)?;
    write_u32_le(mem, out_fd_ptr, 0)?;

    let path_bytes = read_mem(mem, path_ptr, path_len)?;
    let path_str = match validate_path_bytes(path_bytes) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    if path_str.len() > 4096 {
        results[0] = Value::I32(ERRNO_NAMETOOLONG);
        return Ok(());
    }

    let (base_host_path, base_allowed_base, base_allowed_inh) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    let had_trailing_slash = path_str.ends_with('/') || path_str.ends_with('\\');
    let host_path = match resolve_under_base(&base_host_path, &path_str) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    if (rights_base & !base_allowed_inh) != 0 || (rights_inh & !base_allowed_inh) != 0 {
        results[0] = Value::I32(ERRNO_NOTCAPABLE);
        return Ok(());
    }
    if (oflags & OFLAGS_TRUNC) != 0 && (base_allowed_base & RIGHT_PATH_FILESTAT_SET_SIZE) == 0 {
        results[0] = Value::I32(ERRNO_NOTCAPABLE);
        return Ok(());
    }
    if (oflags & OFLAGS_CREAT) != 0 && (base_allowed_base & RIGHT_PATH_CREATE_FILE) == 0 {
        results[0] = Value::I32(ERRNO_NOTCAPABLE);
        return Ok(());
    }

    let nofollow_kind = match stat_path_kind(&host_path, false) {
        Ok(kind) => {
            if kind.is_symlink && (dirflags & 1) == 0 {
                results[0] = Value::I32(if (oflags & OFLAGS_DIRECTORY) != 0 {
                    ERRNO_NOTDIR
                } else {
                    ERRNO_LOOP
                });
                return Ok(());
            }
            Some(kind)
        }
        Err(_) => None,
    };

    if had_trailing_slash {
        match nofollow_kind {
            Some(kind) => {
                if kind.is_file {
                    results[0] = Value::I32(ERRNO_NOTDIR);
                    return Ok(());
                }
            }
            None => match stat_path_kind(&host_path, false) {
                Ok(kind) => {
                    if kind.is_file {
                        results[0] = Value::I32(ERRNO_NOTDIR);
                        return Ok(());
                    }
                }
                Err(e) => {
                    results[0] = Value::I32(path_error_to_errno(&e));
                    return Ok(());
                }
            },
        }
    }

    let follow_kind = match stat_path_kind(&host_path, true) {
        Ok(kind) => Some(kind),
        Err(e) => {
            if (oflags & OFLAGS_DIRECTORY) != 0 {
                results[0] = Value::I32(path_error_to_errno(&e));
                return Ok(());
            }
            None
        }
    };

    if (oflags & OFLAGS_DIRECTORY) != 0 {
        if !follow_kind.is_some_and(|kind| kind.is_dir) {
            results[0] = Value::I32(ERRNO_NOTDIR);
            return Ok(());
        }
    }

    if follow_kind.is_some_and(|kind| kind.is_dir) {
        let has_read = (rights_base & RIGHT_FD_READ) != 0;
        let has_write = (rights_base & RIGHT_FD_WRITE) != 0;
        if has_read && has_write {
            results[0] = Value::I32(ERRNO_ISDIR);
            return Ok(());
        }

        let allowed_dir_base =
            RIGHT_FD_READDIR | RIGHT_FD_FILESTAT_GET | RIGHT_FD_FILESTAT_SET_TIMES | PATH_RIGHTS;
        let dir_rights = rights_base & allowed_dir_base;
        let new_fd = super::with_ctx_mut(|ctx| {
            ctx.alloc_fd(FdEntry::Dir {
                host_path: host_path.clone(),
                rights_base: dir_rights,
                rights_inh,
            })
        });
        write_u32_le(mem, out_fd_ptr, new_fd as u32)?;
        results[0] = Value::I32(ERRNO_SUCCESS);
        return Ok(());
    }

    if follow_kind.is_some_and(|kind| kind.is_file) {
        if (oflags & OFLAGS_CREAT) != 0 && (oflags & OFLAGS_EXCL) != 0 {
            results[0] = Value::I32(ERRNO_EXIST);
            return Ok(());
        }

        let mut opts = std::fs::OpenOptions::new();
        let req_read = (rights_base & RIGHT_FD_READ) != 0;
        let req_write = (rights_base & RIGHT_FD_WRITE) != 0;
        if req_read {
            opts.read(true);
        }
        if req_write {
            opts.write(true);
        }
        if !req_read && !req_write {
            opts.read(true);
        }
        if (oflags & OFLAGS_TRUNC) != 0 {
            opts.truncate(true).write(true);
        }

        match opts.open(&host_path) {
            Ok(file) => {
                let file_rights = rights_base & !PATH_RIGHTS;
                let new_fd = super::with_ctx_mut(|ctx| {
                    ctx.alloc_fd(FdEntry::File {
                        file,
                        host_path: host_path.clone(),
                        rights_base: file_rights,
                        rights_inh,
                        fdflags,
                    })
                });
                write_u32_le(mem, out_fd_ptr, new_fd as u32)?;
                results[0] = Value::I32(ERRNO_SUCCESS);
            }
            Err(e) => {
                let errno = path_error_to_errno(&e);
                results[0] = Value::I32(errno);
            }
        }
        return Ok(());
    }

    if (oflags & OFLAGS_CREAT) != 0 {
        if (oflags & OFLAGS_DIRECTORY) != 0 {
            results[0] = Value::I32(ERRNO_NOENT);
            return Ok(());
        }

        let req_read = (rights_base & RIGHT_FD_READ) != 0;
        let req_write = (rights_base & RIGHT_FD_WRITE) != 0;
        if !req_write {
            let mut precreate = std::fs::OpenOptions::new();
            precreate.write(true);
            if (oflags & OFLAGS_EXCL) != 0 {
                precreate.create_new(true);
            } else {
                precreate.create(true);
            }
            match precreate.open(&host_path) {
                Ok(_) => {}
                Err(e) => match e.kind() {
                    std::io::ErrorKind::AlreadyExists => {}
                    _ => {
                        results[0] = Value::I32(path_error_to_errno(&e));
                        return Ok(());
                    }
                },
            }
        }

        let mut opts = std::fs::OpenOptions::new();
        if req_read {
            opts.read(true);
        }
        if req_write {
            opts.write(true);
        }
        if !req_read && !req_write {
            opts.read(true);
        }
        if req_write {
            if (oflags & OFLAGS_EXCL) != 0 {
                opts.create_new(true);
            } else {
                opts.create(true);
            }
        }
        if (oflags & OFLAGS_TRUNC) != 0 {
            opts.truncate(true).write(true);
        }

        match opts.open(&host_path) {
            Ok(file) => {
                let file_rights = rights_base & !PATH_RIGHTS;
                let new_fd = super::with_ctx_mut(|ctx| {
                    ctx.alloc_fd(FdEntry::File {
                        file,
                        host_path: host_path.clone(),
                        rights_base: file_rights,
                        rights_inh,
                        fdflags,
                    })
                });
                write_u32_le(mem, out_fd_ptr, new_fd as u32)?;
                results[0] = Value::I32(ERRNO_SUCCESS);
            }
            Err(e) => {
                results[0] = Value::I32(path_error_to_errno(&e));
            }
        }
        return Ok(());
    }

    results[0] = Value::I32(ERRNO_NOENT);

    Ok(())
}

// ===========================================================================
// WASI preview1 functions — stubs
// ===========================================================================

pub(crate) fn fd_fdstat_set_flags(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let flags = as_i32(&args[1])? as u16;
    let supported =
        FDFLAGS_APPEND | FDFLAGS_DSYNC | FDFLAGS_NONBLOCK | FDFLAGS_RSYNC | FDFLAGS_SYNC;
    if (flags & !supported) != 0 {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }
    if (0..=2).contains(&fd) {
        let errno = super::with_ctx(|ctx| {
            if ctx.closed_stdio.contains(&fd) {
                ERRNO_BADF
            } else {
                ERRNO_SUCCESS
            }
        });
        results[0] = Value::I32(errno);
        return Ok(());
    }

    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File { fdflags, .. }) => {
            *fdflags = flags;
            ERRNO_SUCCESS
        }
        Some(FdEntry::Dir { .. }) => ERRNO_ISDIR,
        None => ERRNO_BADF,
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_fdstat_set_rights(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let rights_base_new = as_i64(&args[1])? as u64;
    let rights_inh_new = as_i64(&args[2])? as u64;

    if (0..=2).contains(&fd) {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            rights_base,
            rights_inh,
            ..
        })
        | Some(FdEntry::Dir {
            rights_base,
            rights_inh,
            ..
        }) => {
            if (rights_base_new & !*rights_base) != 0 || (rights_inh_new & !*rights_inh) != 0 {
                ERRNO_NOTCAPABLE
            } else {
                *rights_base = rights_base_new;
                *rights_inh = rights_inh_new;
                ERRNO_SUCCESS
            }
        }
        None => ERRNO_BADF,
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_renumber(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let from = as_i32(&args[0])?;
    let to = as_i32(&args[1])?;
    if from == to {
        results[0] = Value::I32(ERRNO_SUCCESS);
        return Ok(());
    }

    let errno = super::with_ctx_mut(|ctx| {
        let dest_valid = if (0..=2).contains(&to) {
            !ctx.closed_stdio.contains(&to)
        } else if preopen_index_for_fd(ctx, to).is_some() {
            true
        } else {
            ctx.fds.contains_key(&to)
        };
        if !dest_valid {
            return ERRNO_BADF;
        }

        let movable = if (0..=2).contains(&from) {
            None
        } else if let Some(idx) = preopen_index_for_fd(ctx, from) {
            Some(FdEntry::Dir {
                host_path: ctx.preopens[idx].host_path.clone(),
                rights_base: RIGHTS_DIR_BASE,
                rights_inh: RIGHTS_DIR_INHERITING,
            })
        } else if let Some(entry) = ctx.fds.remove(&from) {
            Some(entry)
        } else {
            return ERRNO_BADF;
        };

        if !(0..=2).contains(&from) {
            if (0..=2).contains(&to) {
                ctx.closed_stdio.insert(to);
            } else if preopen_index_for_fd(ctx, to).is_some() {
                ctx.closed_preopens.insert(to);
            } else {
                let _ = ctx.fds.remove(&to);
            }
        }

        if let Some(entry) = movable {
            ctx.fds.insert(to, entry);
        }

        if (0..=2).contains(&from) {
            ctx.closed_stdio.insert(from);
        } else if preopen_index_for_fd(ctx, from).is_some() {
            ctx.closed_preopens.insert(from);
        }

        ERRNO_SUCCESS
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_filestat_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let buf_ptr = as_i32(&args[1])? as u32;
    let mem = get_mem(caller)?;

    if (0..=2).contains(&fd) {
        write_filestat(mem, buf_ptr, 1, 2, FILETYPE_CHARACTER_DEVICE, 1, 0, 0, 0, 0)?;
        results[0] = Value::I32(ERRNO_SUCCESS);
        return Ok(());
    }

    enum StatTarget {
        Path(PathBuf),
        Stat(HostStat),
    }

    let target = super::with_ctx(|ctx| {
        if let Some(idx) = preopen_index_for_fd(ctx, fd) {
            return Ok(StatTarget::Path(ctx.preopens[idx].host_path.clone()));
        }
        match ctx.fds.get(&fd) {
            Some(FdEntry::Dir { host_path, .. }) => Ok(StatTarget::Path(host_path.clone())),
            Some(FdEntry::File {
                file, host_path, ..
            }) => match stat_file_metadata(file, host_path) {
                Ok(stat) => Ok(StatTarget::Stat(stat)),
                Err(_) => Err(ERRNO_NOENT),
            },
            None => Err(ERRNO_BADF),
        }
    });

    match target {
        Ok(StatTarget::Path(host_path)) => match stat_path_metadata(&host_path, true) {
            Ok(stat) => {
                write_host_filestat(mem, buf_ptr, stat)?;
                results[0] = Value::I32(ERRNO_SUCCESS);
            }
            Err(_) => results[0] = Value::I32(ERRNO_NOENT),
        },
        Ok(StatTarget::Stat(stat)) => {
            write_host_filestat(mem, buf_ptr, stat)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        Err(errno) => results[0] = Value::I32(errno),
    }
    Ok(())
}

pub(crate) fn fd_filestat_set_size(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let size = as_i64(&args[1])? as u64;

    if (0..=2).contains(&fd) {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_FILESTAT_SET_SIZE) == 0 {
                ERRNO_NOTCAPABLE
            } else if file.set_len(size).is_ok() {
                ERRNO_SUCCESS
            } else {
                ERRNO_IO
            }
        }
        Some(FdEntry::Dir { .. }) => ERRNO_ISDIR,
        None => ERRNO_BADF,
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_filestat_set_times(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    const SET_ATIM: u32 = 1 << 0;
    const SET_ATIM_NOW: u32 = 1 << 1;
    const SET_MTIM: u32 = 1 << 2;
    const SET_MTIM_NOW: u32 = 1 << 3;

    let fd = as_i32(&args[0])?;
    let atim = as_i64(&args[1])? as u64;
    let mtim = as_i64(&args[2])? as u64;
    let fst_flags = as_i32(&args[3])? as u32;
    let invalid_combo = (fst_flags & SET_ATIM != 0 && fst_flags & SET_ATIM_NOW != 0)
        || (fst_flags & SET_MTIM != 0 && fst_flags & SET_MTIM_NOW != 0)
        || (fst_flags & !(SET_ATIM | SET_ATIM_NOW | SET_MTIM | SET_MTIM_NOW) != 0);
    if invalid_combo {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }
    if (0..=2).contains(&fd) {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let errno = super::with_ctx(|ctx| match ctx.fds.get(&fd) {
        Some(FdEntry::File {
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_FILESTAT_SET_TIMES) == 0 {
                return ERRNO_NOTCAPABLE;
            }
            let at_opt = if (fst_flags & SET_ATIM_NOW) != 0 {
                Some(FileTime::now())
            } else if (fst_flags & SET_ATIM) != 0 {
                Some(ns_to_filetime(atim))
            } else {
                None
            };
            let mt_opt = if (fst_flags & SET_MTIM_NOW) != 0 {
                Some(FileTime::now())
            } else if (fst_flags & SET_MTIM) != 0 {
                Some(ns_to_filetime(mtim))
            } else {
                None
            };
            if set_file_handle_times(file, at_opt, mt_opt).is_ok() {
                ERRNO_SUCCESS
            } else {
                ERRNO_IO
            }
        }
        Some(FdEntry::Dir { .. }) => ERRNO_INVAL,
        None => ERRNO_BADF,
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_sync(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    if (0..=2).contains(&fd) {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_SYNC) == 0 {
                ERRNO_NOTCAPABLE
            } else if file.sync_all().is_ok() {
                ERRNO_SUCCESS
            } else {
                ERRNO_IO
            }
        }
        Some(FdEntry::Dir { .. }) => ERRNO_INVAL,
        None => ERRNO_BADF,
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_datasync(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    if (0..=2).contains(&fd) {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_DATASYNC) == 0 {
                ERRNO_NOTCAPABLE
            } else if file.sync_data().is_ok() {
                ERRNO_SUCCESS
            } else {
                ERRNO_IO
            }
        }
        Some(FdEntry::Dir { .. }) => ERRNO_INVAL,
        None => ERRNO_BADF,
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_readdir(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let buf_ptr = as_i32(&args[1])? as u32;
    let buf_len = as_i32(&args[2])? as u32;
    let cookie = as_i64(&args[3])? as u64;
    let used_ptr = as_i32(&args[4])? as u32;
    let mem = get_mem(caller)?;
    write_u32_le(mem, used_ptr, 0)?;

    let host_dir = match dir_fd_state(fd) {
        Ok((host, _, _)) => host,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    let read_dir = match std::fs::read_dir(&host_dir) {
        Ok(rd) => rd,
        Err(e) => {
            results[0] = Value::I32(path_error_to_errno(&e));
            return Ok(());
        }
    };

    let mut entries: Vec<(String, u8, u64)> = Vec::new();
    entries.push((
        ".".to_string(),
        FILETYPE_DIRECTORY,
        derive_ino_for_path(&host_dir),
    ));
    entries.push((
        "..".to_string(),
        FILETYPE_DIRECTORY,
        host_dir.parent().map(derive_ino_for_path).unwrap_or(0),
    ));
    let mut rest = Vec::new();
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let filetype = match entry.file_type() {
            Ok(ft) => {
                if ft.is_dir() {
                    FILETYPE_DIRECTORY
                } else if ft.is_file() {
                    FILETYPE_REGULAR_FILE
                } else if ft.is_symlink() {
                    FILETYPE_SYMBOLIC_LINK
                } else {
                    FILETYPE_UNKNOWN
                }
            }
            Err(_) => FILETYPE_UNKNOWN,
        };
        let child_host = host_dir.join(&name);
        rest.push((name, filetype, derive_ino_for_path(&child_host)));
    }
    rest.sort_by(|a, b| a.0.cmp(&b.0));
    entries.extend(rest);

    let mut idx = cookie as usize;
    if idx > entries.len() {
        idx = entries.len();
    }

    let mut used: u32 = 0;
    let mut write_off = buf_ptr;
    while idx < entries.len() {
        let (name, ftype, ino) = &entries[idx];
        let name_bytes = name.as_bytes();
        let record_len = 24 + name_bytes.len() as u32;
        if used + record_len > buf_len {
            break;
        }
        write_u64_le(mem, write_off, (idx as u64) + 1)?;
        write_u64_le(mem, write_off + 8, *ino)?;
        write_u32_le(mem, write_off + 16, name_bytes.len() as u32)?;
        write_u8_le(mem, write_off + 20, *ftype)?;
        write_mem(mem, write_off + 24, name_bytes)?;
        write_off += record_len;
        used += record_len;
        idx += 1;
    }

    write_u32_le(
        mem,
        used_ptr,
        if idx < entries.len() { buf_len } else { used },
    )?;
    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

pub(crate) fn fd_pread(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let iovs_ptr = as_i32(&args[1])? as u32;
    let iovs_len = as_i32(&args[2])? as u32;
    let offset = as_i64(&args[3])? as u64;
    let out_ptr = as_i32(&args[4])? as u32;
    let mem = get_mem(caller)?;
    write_u32_le(mem, out_ptr, 0)?;

    let mut total: u32 = 0;
    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_READ) == 0 {
                return ERRNO_BADF;
            }
            let cur = file.stream_position().ok();
            if file.seek(SeekFrom::Start(offset)).is_err() {
                return ERRNO_IO;
            }
            for i in 0..iovs_len {
                let base_ptr = iovs_ptr + i * 8;
                let ptr = match read_u32_le(mem, base_ptr) {
                    Ok(v) => v,
                    Err(_) => return ERRNO_INVAL,
                };
                let len = match read_u32_le(mem, base_ptr + 4) {
                    Ok(v) => v,
                    Err(_) => return ERRNO_INVAL,
                };
                if len == 0 {
                    continue;
                }
                let start = ptr as usize;
                let end = start + len as usize;
                if end > mem.len() {
                    return ERRNO_INVAL;
                }
                match file.read(&mut mem[start..end]) {
                    Ok(0) => break,
                    Ok(n) => {
                        total = total.saturating_add(n as u32);
                        if n < len as usize {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            if let Some(pos) = cur {
                let _ = file.seek(SeekFrom::Start(pos));
            }
            ERRNO_SUCCESS
        }
        Some(FdEntry::Dir { .. }) => ERRNO_ISDIR,
        None => ERRNO_BADF,
    });

    write_u32_le(mem, out_ptr, total)?;
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_pwrite(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let iovs_ptr = as_i32(&args[1])? as u32;
    let iovs_len = as_i32(&args[2])? as u32;
    let offset = as_i64(&args[3])? as u64;
    let out_ptr = as_i32(&args[4])? as u32;
    let mem = get_mem(caller)?;
    write_u32_le(mem, out_ptr, 0)?;

    let mut total: u32 = 0;
    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_WRITE) == 0 {
                return ERRNO_BADF;
            }
            let cur = file.stream_position().ok();
            if file.seek(SeekFrom::Start(offset)).is_err() {
                return ERRNO_IO;
            }
            for i in 0..iovs_len {
                let base_ptr = iovs_ptr + i * 8;
                let ptr = match read_u32_le(mem, base_ptr) {
                    Ok(v) => v,
                    Err(_) => return ERRNO_INVAL,
                };
                let len = match read_u32_le(mem, base_ptr + 4) {
                    Ok(v) => v,
                    Err(_) => return ERRNO_INVAL,
                };
                if len == 0 {
                    continue;
                }
                let bytes = match read_mem(mem, ptr, len) {
                    Ok(v) => v,
                    Err(_) => return ERRNO_INVAL,
                };
                if file.write_all(bytes).is_err() {
                    break;
                }
                total = total.saturating_add(len);
            }
            if let Some(pos) = cur {
                let _ = file.seek(SeekFrom::Start(pos));
            }
            ERRNO_SUCCESS
        }
        Some(FdEntry::Dir { .. }) => ERRNO_ISDIR,
        None => ERRNO_BADF,
    });

    write_u32_le(mem, out_ptr, total)?;
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_allocate(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let offset = as_i64(&args[1])? as u64;
    let len = as_i64(&args[2])? as u64;

    if (0..=2).contains(&fd) {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let errno = super::with_ctx_mut(|ctx| match ctx.fds.get_mut(&fd) {
        Some(FdEntry::File {
            file,
            host_path,
            rights_base,
            ..
        }) => {
            if (*rights_base & RIGHT_FD_ALLOCATE) == 0 {
                return ERRNO_NOTCAPABLE;
            }
            let want = offset.saturating_add(len);
            match stat_file_metadata(file, host_path) {
                Ok(stat) => {
                    if want > stat.size && file.set_len(want).is_err() {
                        ERRNO_IO
                    } else {
                        ERRNO_SUCCESS
                    }
                }
                Err(_) => ERRNO_IO,
            }
        }
        Some(FdEntry::Dir { .. }) => ERRNO_ISDIR,
        None => ERRNO_BADF,
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn fd_advise(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    results[0] = Value::I32(if (0..=2).contains(&fd) {
        ERRNO_INVAL
    } else {
        ERRNO_SUCCESS
    });
    Ok(())
}

pub(crate) fn sched_yield(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

pub(crate) fn sock_shutdown(
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let _how = as_i32(&args[1])?;
    let errno = super::with_ctx(|ctx| {
        if (0..=2).contains(&fd) {
            return if ctx.closed_stdio.contains(&fd) {
                ERRNO_BADF
            } else {
                ERRNO_NOTSOCK
            };
        }
        if ctx.fds.contains_key(&fd) {
            ERRNO_NOTSOCK
        } else {
            ERRNO_BADF
        }
    });
    results[0] = Value::I32(errno);
    Ok(())
}

pub(crate) fn poll_oneoff(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let in_ptr = as_i32(&args[0])? as u32;
    let out_ptr = as_i32(&args[1])? as u32;
    let nsubs = as_i32(&args[2])? as u32;
    let out_nready_ptr = as_i32(&args[3])? as u32;
    let mem = get_mem(caller)?;

    struct Sub {
        userdata: [u8; 8],
        tag: u8,
        fd: Option<i32>,
    }

    let mut subs = Vec::with_capacity(nsubs as usize);
    for i in 0..nsubs {
        let sub_off = in_ptr + i * 48;
        let mut userdata = [0u8; 8];
        userdata.copy_from_slice(read_mem(mem, sub_off, 8)?);
        let tag = read_mem(mem, sub_off + 8, 1)?[0];
        let fd = if tag == 1 || tag == 2 {
            Some(read_u32_le(mem, sub_off + 16)? as i32)
        } else {
            None
        };
        subs.push(Sub { userdata, tag, fd });
    }

    let mut ready_indices = Vec::new();
    let mut clock_index = None;
    for (i, sub) in subs.iter().enumerate() {
        match sub.tag {
            0 => {
                if clock_index.is_none() {
                    clock_index = Some(i);
                }
            }
            2 => {
                if matches!(sub.fd, Some(1 | 2)) {
                    ready_indices.push(i);
                }
            }
            _ => {}
        }
    }
    if ready_indices.is_empty() {
        if let Some(idx) = clock_index {
            ready_indices.push(idx);
        }
    }
    ready_indices.sort_unstable();

    for (out_i, &sub_i) in ready_indices.iter().enumerate() {
        let sub = &subs[sub_i];
        let mut out_event = [0u8; 32];
        out_event[0..8].copy_from_slice(&sub.userdata);
        out_event[8..10].copy_from_slice(&(ERRNO_SUCCESS as u16).to_le_bytes());
        out_event[10] = sub.tag;
        write_mem(mem, out_ptr + (out_i as u32) * 32, &out_event)?;
    }
    write_u32_le(mem, out_nready_ptr, ready_indices.len() as u32)?;
    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

pub(crate) fn path_create_directory(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let path_ptr = as_i32(&args[1])? as u32;
    let path_len = as_i32(&args[2])? as u32;
    let mem = get_mem(caller)?;

    let (base_host, _, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let rel = match validate_path_bytes(read_mem(mem, path_ptr, path_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let host_path = match resolve_under_base(&base_host, &rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    results[0] = Value::I32(match std::fs::create_dir(&host_path) {
        Ok(()) => ERRNO_SUCCESS,
        Err(e) => path_error_to_errno(&e),
    });
    Ok(())
}

pub(crate) fn path_filestat_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let flags = as_i32(&args[1])? as u32;
    let path_ptr = as_i32(&args[2])? as u32;
    let path_len = as_i32(&args[3])? as u32;
    let buf_ptr = as_i32(&args[4])? as u32;
    let mem = get_mem(caller)?;
    let (base_host, _, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let rel = match validate_path_bytes(read_mem(mem, path_ptr, path_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let host_path = match resolve_under_base(&base_host, &rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    let stat = match stat_path_metadata(&host_path, (flags & 1) != 0) {
        Ok(stat) => stat,
        Err(e) => {
            results[0] = Value::I32(path_error_to_errno(&e));
            return Ok(());
        }
    };
    write_host_filestat(mem, buf_ptr, stat)?;
    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

pub(crate) fn path_filestat_set_times(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    const SET_ATIM: u32 = 1 << 0;
    const SET_ATIM_NOW: u32 = 1 << 1;
    const SET_MTIM: u32 = 1 << 2;
    const SET_MTIM_NOW: u32 = 1 << 3;

    let fd = as_i32(&args[0])?;
    let flags = as_i32(&args[1])? as u32;
    let path_ptr = as_i32(&args[2])? as u32;
    let path_len = as_i32(&args[3])? as u32;
    let atim = as_i64(&args[4])? as u64;
    let mtim = as_i64(&args[5])? as u64;
    let fst_flags = as_i32(&args[6])? as u32;
    let invalid_combo = (fst_flags & SET_ATIM != 0 && fst_flags & SET_ATIM_NOW != 0)
        || (fst_flags & SET_MTIM != 0 && fst_flags & SET_MTIM_NOW != 0)
        || (fst_flags & !(SET_ATIM | SET_ATIM_NOW | SET_MTIM | SET_MTIM_NOW) != 0);
    if invalid_combo {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let mem = get_mem(caller)?;
    let (base_host, base_rights, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    if (base_rights & RIGHT_PATH_FILESTAT_SET_TIMES) == 0 {
        results[0] = Value::I32(ERRNO_NOTCAPABLE);
        return Ok(());
    }
    let rel = match validate_path_bytes(read_mem(mem, path_ptr, path_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let host_path = match resolve_under_base(&base_host, &rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    if let Err(e) = stat_path_metadata(&host_path, false) {
        results[0] = Value::I32(path_error_to_errno(&e));
        return Ok(());
    }

    let at_opt = if (fst_flags & SET_ATIM_NOW) != 0 {
        Some(FileTime::now())
    } else if (fst_flags & SET_ATIM) != 0 {
        Some(ns_to_filetime(atim))
    } else {
        None
    };
    let mt_opt = if (fst_flags & SET_MTIM_NOW) != 0 {
        Some(FileTime::now())
    } else if (fst_flags & SET_MTIM) != 0 {
        Some(ns_to_filetime(mtim))
    } else {
        None
    };
    let result = if (flags & 1) != 0 {
        set_file_times(
            &host_path,
            at_opt.unwrap_or_else(FileTime::now),
            mt_opt.unwrap_or_else(FileTime::now),
        )
    } else {
        set_symlink_file_times(
            &host_path,
            at_opt.unwrap_or_else(FileTime::now),
            mt_opt.unwrap_or_else(FileTime::now),
        )
    };
    results[0] = Value::I32(if result.is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_IO
    });
    Ok(())
}

pub(crate) fn path_readlink(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let path_ptr = as_i32(&args[1])? as u32;
    let path_len = as_i32(&args[2])? as u32;
    let buf_ptr = as_i32(&args[3])? as u32;
    let buf_len = as_i32(&args[4])? as u32;
    let out_len_ptr = as_i32(&args[5])? as u32;
    let mem = get_mem(caller)?;
    write_u32_le(mem, out_len_ptr, 0)?;

    let (base_host, _, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let rel = match validate_path_bytes(read_mem(mem, path_ptr, path_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let rel_trimmed = rel.trim_end_matches(['/', '\\']).to_string();
    let host = match resolve_under_base(&base_host, &rel_trimmed) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    match std::fs::read_link(&host) {
        Ok(target) => {
            if target.is_absolute() {
                results[0] = Value::I32(ERRNO_PERM);
                return Ok(());
            }
            let text = target.to_string_lossy();
            let bytes = text.as_bytes();
            let n = bytes.len().min(buf_len as usize);
            write_mem(mem, buf_ptr, &bytes[..n])?;
            write_u32_le(mem, out_len_ptr, n as u32)?;
            results[0] = Value::I32(ERRNO_SUCCESS);
        }
        Err(e) => results[0] = Value::I32(path_error_to_errno(&e)),
    }
    Ok(())
}

pub(crate) fn path_remove_directory(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let path_ptr = as_i32(&args[1])? as u32;
    let path_len = as_i32(&args[2])? as u32;
    let mem = get_mem(caller)?;

    let (base_host, _, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let rel = match validate_path_bytes(read_mem(mem, path_ptr, path_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let has_trailing_slash = rel.ends_with('/') || rel.ends_with('\\');
    let rel_trimmed = rel.trim_end_matches(['/', '\\']).to_string();
    let host = match resolve_under_base(&base_host, &rel_trimmed) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    if has_trailing_slash {
        match stat_path_metadata(&host, false) {
            Ok(stat) => {
                if stat.kind.is_file {
                    results[0] = Value::I32(if cfg!(windows) {
                        ERRNO_NOENT
                    } else {
                        ERRNO_NOTDIR
                    });
                    return Ok(());
                }
            }
            Err(e) => {
                results[0] = Value::I32(path_error_to_errno(&e));
                return Ok(());
            }
        }
    }

    results[0] = Value::I32(match std::fs::remove_dir(&host) {
        Ok(()) => ERRNO_SUCCESS,
        Err(e) => path_error_to_errno(&e),
    });
    Ok(())
}

pub(crate) fn path_unlink_file(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let path_ptr = as_i32(&args[1])? as u32;
    let path_len = as_i32(&args[2])? as u32;
    let mem = get_mem(caller)?;

    let (base_host, _, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let rel = match validate_path_bytes(read_mem(mem, path_ptr, path_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let has_trailing_slash = rel.ends_with('/') || rel.ends_with('\\');
    let rel_trimmed = rel.trim_end_matches(['/', '\\']).to_string();
    let host = match resolve_under_base(&base_host, &rel_trimmed) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    if has_trailing_slash {
        match stat_path_metadata(&host, false) {
            Ok(stat) => {
                results[0] = Value::I32(if stat.kind.is_dir {
                    ERRNO_ISDIR
                } else {
                    ERRNO_NOTDIR
                });
                return Ok(());
            }
            Err(e) => {
                results[0] = Value::I32(path_error_to_errno(&e));
                return Ok(());
            }
        }
    }

    if let Ok(stat) = stat_path_metadata(&host, false) {
        if stat.kind.is_symlink {
            results[0] = Value::I32(match std::fs::remove_file(&host) {
                Ok(()) => ERRNO_SUCCESS,
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::IsADirectory
                        || e.kind() == std::io::ErrorKind::PermissionDenied
                    {
                        match std::fs::remove_dir(&host) {
                            Ok(()) => ERRNO_SUCCESS,
                            Err(e2) => path_error_to_errno(&e2),
                        }
                    } else {
                        path_error_to_errno(&e)
                    }
                }
            });
            return Ok(());
        }
    }

    results[0] = Value::I32(match std::fs::remove_file(&host) {
        Ok(()) => ERRNO_SUCCESS,
        Err(e) => path_error_to_errno(&e),
    });
    Ok(())
}

pub(crate) fn path_rename(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let old_ptr = as_i32(&args[1])? as u32;
    let old_len = as_i32(&args[2])? as u32;
    let new_fd = as_i32(&args[3])?;
    let new_ptr = as_i32(&args[4])? as u32;
    let new_len = as_i32(&args[5])? as u32;
    let mem = get_mem(caller)?;

    let (base_old, _, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let (base_new, _, _) = match dir_fd_state(new_fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let old_rel = match validate_path_bytes(read_mem(mem, old_ptr, old_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let new_rel = match validate_path_bytes(read_mem(mem, new_ptr, new_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let old_host = match resolve_under_base(&base_old, &old_rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let new_host = match resolve_under_base(&base_new, &new_rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    if let (Ok(old_stat), Ok(new_stat)) = (
        stat_path_metadata(&old_host, false),
        stat_path_metadata(&new_host, false),
    ) {
        if old_stat.kind.is_dir && new_stat.kind.is_file {
            results[0] = Value::I32(ERRNO_NOTDIR);
            return Ok(());
        }
        if old_stat.kind.is_file && new_stat.kind.is_dir {
            results[0] = Value::I32(ERRNO_ISDIR);
            return Ok(());
        }
    }

    results[0] = Value::I32(match std::fs::rename(&old_host, &new_host) {
        Ok(()) => ERRNO_SUCCESS,
        Err(e) => path_error_to_errno(&e),
    });
    Ok(())
}

pub(crate) fn path_link(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let old_fd = as_i32(&args[0])?;
    let old_flags = as_i32(&args[1])? as u32;
    let old_ptr = as_i32(&args[2])? as u32;
    let old_len = as_i32(&args[3])? as u32;
    let new_fd = as_i32(&args[4])?;
    let new_ptr = as_i32(&args[5])? as u32;
    let new_len = as_i32(&args[6])? as u32;
    let mem = get_mem(caller)?;

    if (old_flags & 1) != 0 {
        results[0] = Value::I32(ERRNO_INVAL);
        return Ok(());
    }

    let (base_old, _, _) = match dir_fd_state(old_fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let (base_new, _, _) = match dir_fd_state(new_fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let old_rel = match validate_path_bytes(read_mem(mem, old_ptr, old_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let new_rel = match validate_path_bytes(read_mem(mem, new_ptr, new_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    if new_rel.ends_with('/') || new_rel.ends_with('\\') {
        results[0] = Value::I32(ERRNO_NOENT);
        return Ok(());
    }
    let old_host = match resolve_under_base(&base_old, &old_rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let new_host = match resolve_under_base(&base_new, &new_rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    if stat_path_metadata(&old_host, false)
        .map(|stat| stat.kind.is_dir)
        .unwrap_or(false)
    {
        results[0] = Value::I32(ERRNO_ACCES);
        return Ok(());
    }

    results[0] = Value::I32(match hard_link_no_follow(&old_host, &new_host) {
        Ok(()) => ERRNO_SUCCESS,
        Err(e) => path_error_to_errno(&e),
    });
    Ok(())
}

pub(crate) fn path_symlink(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let old_ptr = as_i32(&args[0])? as u32;
    let old_len = as_i32(&args[1])? as u32;
    let fd = as_i32(&args[2])?;
    let new_ptr = as_i32(&args[3])? as u32;
    let new_len = as_i32(&args[4])? as u32;
    let mem = get_mem(caller)?;

    let old = match validate_path_bytes(read_mem(mem, old_ptr, old_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let (base, _, _) = match dir_fd_state(fd) {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };
    let new_rel = match validate_path_bytes(read_mem(mem, new_ptr, new_len)?) {
        Ok(s) => s,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    if new_rel.ends_with('/') || new_rel.ends_with('\\') {
        #[cfg(windows)]
        {
            results[0] = Value::I32(ERRNO_NOENT);
            return Ok(());
        }
        #[cfg(not(windows))]
        {
            let trimmed = new_rel.trim_end_matches(['/', '\\']);
            let candidate = match resolve_under_base(&base, trimmed) {
                Ok(p) => p,
                Err(_) => {
                    results[0] = Value::I32(ERRNO_NOENT);
                    return Ok(());
                }
            };
            results[0] = Value::I32(match stat_path_metadata(&candidate, false) {
                Ok(stat) => {
                    if stat.kind.is_dir {
                        ERRNO_EXIST
                    } else if stat.kind.is_file {
                        ERRNO_NOTDIR
                    } else {
                        ERRNO_EXIST
                    }
                }
                Err(_) => ERRNO_NOENT,
            });
            return Ok(());
        }
    }

    let old_abs = {
        let p = Path::new(&old);
        p.is_absolute() || old.starts_with('/') || old.starts_with('\\')
    };
    if old_abs {
        results[0] = Value::I32(ERRNO_PERM);
        return Ok(());
    }

    let new_host = match resolve_under_base(&base, &new_rel) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    #[cfg(windows)]
    {
        if std::fs::symlink_metadata(&new_host).is_ok() {
            results[0] = Value::I32(ERRNO_NOENT);
            return Ok(());
        }
        let old_norm = PathBuf::from(old.replace('/', "\\"));
        let is_dir = std::fs::symlink_metadata(Path::new(&base).join(&old_norm))
            .map(|m| m.is_dir())
            .unwrap_or(false);
        let res = if is_dir {
            std::os::windows::fs::symlink_dir(&old_norm, &new_host)
        } else {
            std::os::windows::fs::symlink_file(&old_norm, &new_host)
        };
        results[0] = Value::I32(match res {
            Ok(()) => ERRNO_SUCCESS,
            Err(e) => match e.kind() {
                std::io::ErrorKind::AlreadyExists
                | std::io::ErrorKind::PermissionDenied
                | std::io::ErrorKind::NotFound => ERRNO_NOENT,
                _ => ERRNO_IO,
            },
        });
    }

    #[cfg(not(windows))]
    {
        results[0] = Value::I32(match std::os::unix::fs::symlink(&old, &new_host) {
            Ok(()) => ERRNO_SUCCESS,
            Err(e) => path_error_to_errno(&e),
        });
    }
    Ok(())
}
