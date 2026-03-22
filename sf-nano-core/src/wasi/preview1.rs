//! WASI preview1 function implementations for sf-nano-core.
//!
//! Each public function has the signature `ExternalFn`:
//! `fn(&mut Caller, &[Value], &mut [Value]) -> Result<(), WasmError>`

use std::format;
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::time::{SystemTime, UNIX_EPOCH};
use std::vec;
use std::vec::Vec;

use crate::error::WasmError;
use crate::vm::entities::Caller;
use crate::vm::value::Value;

use super::FdEntry;

// ---------------------------------------------------------------------------
// WASI errno constants
// ---------------------------------------------------------------------------

const ERRNO_SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8;
const ERRNO_ILSEQ: i32 = 25;
const ERRNO_INVAL: i32 = 28;
const ERRNO_IO: i32 = 29;
const ERRNO_ISDIR: i32 = 31;
const ERRNO_NAMETOOLONG: i32 = 37;
const ERRNO_NOENT: i32 = 44;
const ERRNO_NOSYS: i32 = 52;
const ERRNO_NOTDIR: i32 = 54;
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
const RIGHT_SOCK_SHUTDOWN: u64 = 1 << 28;

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

// WASI filetype constants
const FILETYPE_UNKNOWN: u8 = 0;
const FILETYPE_CHARACTER_DEVICE: u8 = 2;
const FILETYPE_DIRECTORY: u8 = 3;
const FILETYPE_REGULAR_FILE: u8 = 4;

// ---------------------------------------------------------------------------
// Memory helper functions
// ---------------------------------------------------------------------------

fn as_i32(v: &Value) -> Result<i32, WasmError> {
    match v {
        Value::I32(n) => Ok(*n),
        _ => Err(WasmError::trap("expected i32 argument".into())),
    }
}

fn as_i64(v: &Value) -> Result<i64, WasmError> {
    match v {
        Value::I64(n) => Ok(*n),
        _ => Err(WasmError::trap("expected i64 argument".into())),
    }
}

fn read_mem(mem: &[u8], ptr: u32, len: u32) -> Result<&[u8], WasmError> {
    let start = ptr as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| WasmError::trap("memory access out of bounds".into()))?;
    if end > mem.len() {
        return Err(WasmError::trap("memory access out of bounds".into()));
    }
    Ok(&mem[start..end])
}

fn write_mem(mem: &mut [u8], ptr: u32, data: &[u8]) -> Result<(), WasmError> {
    let start = ptr as usize;
    let end = start
        .checked_add(data.len())
        .ok_or_else(|| WasmError::trap("memory access out of bounds".into()))?;
    if end > mem.len() {
        return Err(WasmError::trap("memory access out of bounds".into()));
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
        .ok_or_else(|| WasmError::trap("no linear memory available".into()))
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
    let mut parts = Vec::new();
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

// ---------------------------------------------------------------------------
// Map std::io::ErrorKind to WASI errno
// ---------------------------------------------------------------------------

fn io_error_to_errno(e: &std::io::Error) -> i32 {
    match e.kind() {
        std::io::ErrorKind::NotFound => ERRNO_NOENT,
        std::io::ErrorKind::PermissionDenied => ERRNO_NOTCAPABLE,
        std::io::ErrorKind::AlreadyExists => 20, // ERRNO_EXIST
        std::io::ErrorKind::InvalidInput => ERRNO_INVAL,
        _ => ERRNO_IO,
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

    let argv: Vec<String> = super::with_ctx(|ctx| ctx.args.clone());

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

    let env: Vec<(String, String)> = super::with_ctx(|ctx| ctx.env.clone());

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
            let mut out_buf = Vec::new();
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
            let mut buffers = Vec::new();
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
            let mut iovs = Vec::new();
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
            let mut iovs = Vec::new();
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
        // preopened dir
        let preopen_end = 3 + ctx.preopens.len() as i32;
        if fd >= 3 && fd < preopen_end {
            if ctx.closed_preopens.contains(&fd) {
                return ERRNO_BADF;
            }
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
            file, rights_base, ..
        }) => {
            if (*rights_base & RIGHT_FD_SEEK) == 0 {
                return Err(ERRNO_NOTCAPABLE);
            }
            match file.seek(seek_from) {
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
            let copy_len = std::cmp::min(bytes.len(), path_len as usize);
            let mem = get_mem(caller)?;
            write_mem(mem, path_ptr, &bytes[..copy_len])?;
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
    let _clock_id = as_i32(&args[0])?;
    let _precision = as_i64(&args[1])?;
    let time_ptr = as_i32(&args[2])? as u32;

    let nanos = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() * 1_000_000_000 + d.subsec_nanos() as u64,
        Err(_) => 0,
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
        return Err(WasmError::trap("memory access out of bounds".into()));
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
    let _dirflags = as_i32(&args[1])?;
    let path_ptr = as_i32(&args[2])? as u32;
    let path_len = as_i32(&args[3])? as u32;
    let oflags = as_i32(&args[4])?;
    let rights_base = as_i64(&args[5])? as u64;
    let rights_inh = as_i64(&args[6])? as u64;
    let fdflags = as_i32(&args[7])? as u16;
    let out_fd_ptr = as_i32(&args[8])? as u32;

    let mem = get_mem(caller)?;

    // Read path from guest memory
    let path_bytes = read_mem(mem, path_ptr, path_len)?;

    // Validate: no NUL bytes
    if path_bytes.contains(&0) {
        results[0] = Value::I32(ERRNO_ILSEQ);
        return Ok(());
    }

    // Validate: valid UTF-8
    let path_str = match std::str::from_utf8(path_bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            results[0] = Value::I32(ERRNO_ILSEQ);
            return Ok(());
        }
    };

    if path_str.len() > 4096 {
        results[0] = Value::I32(ERRNO_NAMETOOLONG);
        return Ok(());
    }

    // Get the base directory info from preopen or dynamic fd
    let base_info: Result<(PathBuf, u64), i32> = super::with_ctx(|ctx| {
        // Check if it's a preopen fd
        let preopen_end = 3 + ctx.preopens.len() as i32;
        if fd >= 3 && fd < preopen_end {
            if ctx.closed_preopens.contains(&fd) {
                return Err(ERRNO_BADF);
            }
            let idx = (fd - 3) as usize;
            return Ok((ctx.preopens[idx].host_path.clone(), RIGHTS_DIR_INHERITING));
        }
        // Check dynamic fds
        match ctx.fds.get(&fd) {
            Some(FdEntry::Dir {
                host_path,
                rights_inh: inh,
                ..
            }) => Ok((host_path.clone(), *inh)),
            Some(FdEntry::File { .. }) => Err(ERRNO_NOTDIR),
            None => Err(ERRNO_BADF),
        }
    });

    let (base_host_path, base_allowed_inh) = match base_info {
        Ok(v) => v,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    // Rights enforcement: requested rights must be a subset of inheriting rights
    if (rights_base & !base_allowed_inh) != 0 || (rights_inh & !base_allowed_inh) != 0 {
        results[0] = Value::I32(ERRNO_NOTCAPABLE);
        return Ok(());
    }

    // Capability checks for specific oflags
    if (oflags & OFLAGS_TRUNC) != 0 && (base_allowed_inh & RIGHT_PATH_FILESTAT_SET_SIZE) == 0 {
        results[0] = Value::I32(ERRNO_NOTCAPABLE);
        return Ok(());
    }
    if (oflags & OFLAGS_CREAT) != 0 && (base_allowed_inh & RIGHT_PATH_CREATE_FILE) == 0 {
        results[0] = Value::I32(ERRNO_NOTCAPABLE);
        return Ok(());
    }

    // Resolve the path
    let host_path = match resolve_under_base(&base_host_path, &path_str) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    let trailing_slash = path_str.ends_with('/');

    // Check if path is a directory
    let is_dir = host_path.is_dir();
    let exists = host_path.exists();

    // DIRECTORY flag handling
    if (oflags & OFLAGS_DIRECTORY) != 0 {
        if exists && !is_dir {
            results[0] = Value::I32(ERRNO_NOTDIR);
            return Ok(());
        }
        if !exists {
            results[0] = Value::I32(ERRNO_NOENT);
            return Ok(());
        }
    }

    // Trailing slash on a non-directory
    if trailing_slash && exists && !is_dir {
        results[0] = Value::I32(ERRNO_NOTDIR);
        return Ok(());
    }

    // Handle opening a directory
    if is_dir {
        // Cannot open directory with both read and write
        if (rights_base & RIGHT_FD_READ) != 0 && (rights_base & RIGHT_FD_WRITE) != 0 {
            results[0] = Value::I32(ERRNO_ISDIR);
            return Ok(());
        }

        // Strip non-path rights, keep path rights for dirs
        let dir_rights = rights_base & (RIGHTS_DIR_BASE | RIGHTS_FILE_BASE);
        let dir_inh = rights_inh;

        let new_fd = super::with_ctx_mut(|ctx| {
            ctx.alloc_fd(FdEntry::Dir {
                host_path: host_path.clone(),
                rights_base: dir_rights,
                rights_inh: dir_inh,
            })
        });

        write_u32_le(mem, out_fd_ptr, new_fd as u32)?;
        results[0] = Value::I32(ERRNO_SUCCESS);
        return Ok(());
    }

    // Handle opening a file
    if exists {
        // CREAT | EXCL with existing file → error
        if (oflags & OFLAGS_CREAT) != 0 && (oflags & OFLAGS_EXCL) != 0 {
            results[0] = Value::I32(20); // ERRNO_EXIST
            return Ok(());
        }

        let mut opts = std::fs::OpenOptions::new();
        if (rights_base & RIGHT_FD_READ) != 0 {
            opts.read(true);
        }
        if (rights_base & RIGHT_FD_WRITE) != 0 {
            opts.write(true);
        }
        if (oflags & OFLAGS_TRUNC) != 0 {
            opts.truncate(true);
            // truncate requires write
            opts.write(true);
        }
        if (fdflags & FDFLAGS_APPEND) != 0 {
            opts.append(true);
        }

        match opts.open(&host_path) {
            Ok(file) => {
                // Sanitize rights: strip PATH_* bits for file fds
                let file_rights = rights_base & RIGHTS_FILE_BASE;
                let new_fd = super::with_ctx_mut(|ctx| {
                    ctx.alloc_fd(FdEntry::File {
                        file,
                        host_path: host_path.clone(),
                        rights_base: file_rights,
                        rights_inh: rights_inh & RIGHTS_FILE_BASE,
                        fdflags,
                    })
                });
                write_u32_le(mem, out_fd_ptr, new_fd as u32)?;
                results[0] = Value::I32(ERRNO_SUCCESS);
            }
            Err(e) => {
                results[0] = Value::I32(io_error_to_errno(&e));
            }
        }
    } else if (oflags & OFLAGS_CREAT) != 0 {
        // File doesn't exist, but CREATE flag is set
        let mut opts = std::fs::OpenOptions::new();
        opts.create(true);
        if (rights_base & RIGHT_FD_READ) != 0 {
            opts.read(true);
        }
        if (rights_base & RIGHT_FD_WRITE) != 0 {
            opts.write(true);
        } else {
            // create requires at least write
            opts.write(true);
        }
        if (oflags & OFLAGS_EXCL) != 0 {
            opts.create_new(true);
        }
        if (fdflags & FDFLAGS_APPEND) != 0 {
            opts.append(true);
        }

        match opts.open(&host_path) {
            Ok(file) => {
                let file_rights = rights_base & RIGHTS_FILE_BASE;
                let new_fd = super::with_ctx_mut(|ctx| {
                    ctx.alloc_fd(FdEntry::File {
                        file,
                        host_path: host_path.clone(),
                        rights_base: file_rights,
                        rights_inh: rights_inh & RIGHTS_FILE_BASE,
                        fdflags,
                    })
                });
                write_u32_le(mem, out_fd_ptr, new_fd as u32)?;
                results[0] = Value::I32(ERRNO_SUCCESS);
            }
            Err(e) => {
                results[0] = Value::I32(io_error_to_errno(&e));
            }
        }
    } else {
        // File doesn't exist and no CREATE flag
        results[0] = Value::I32(ERRNO_NOENT);
    }

    Ok(())
}

// ===========================================================================
// WASI preview1 functions — stubs
// ===========================================================================

pub(crate) fn fd_fdstat_set_flags(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_fdstat_set_rights(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_renumber(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_filestat_get(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_filestat_set_size(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_filestat_set_times(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_sync(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_datasync(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_readdir(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_pread(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_pwrite(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_allocate(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn fd_advise(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
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
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn poll_oneoff(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_create_directory(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_filestat_get(
    caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let fd = as_i32(&args[0])?;
    let _flags = as_i32(&args[1])?; // lookup flags (e.g. symlink follow)
    let path_ptr = as_i32(&args[2])? as u32;
    let path_len = as_i32(&args[3])? as u32;
    let buf_ptr = as_i32(&args[4])? as u32;

    let mem = get_mem(caller)?;

    // Read path from guest memory
    let path_bytes = read_mem(mem, path_ptr, path_len)?;
    let path_str = match std::str::from_utf8(path_bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            results[0] = Value::I32(ERRNO_ILSEQ);
            return Ok(());
        }
    };

    // Resolve the base directory from the fd
    let base_path: Result<PathBuf, i32> = super::with_ctx(|ctx| {
        let preopen_end = 3 + ctx.preopens.len() as i32;
        if fd >= 3 && fd < preopen_end {
            if ctx.closed_preopens.contains(&fd) {
                return Err(ERRNO_BADF);
            }
            let idx = (fd - 3) as usize;
            return Ok(ctx.preopens[idx].host_path.clone());
        }
        match ctx.fds.get(&fd) {
            Some(FdEntry::Dir { host_path, .. }) => Ok(host_path.clone()),
            Some(FdEntry::File { .. }) => Err(ERRNO_NOTDIR),
            None => Err(ERRNO_BADF),
        }
    });

    let base = match base_path {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    let host_path = match resolve_under_base(&base, &path_str) {
        Ok(p) => p,
        Err(errno) => {
            results[0] = Value::I32(errno);
            return Ok(());
        }
    };

    // Get metadata from the host filesystem
    let metadata = match std::fs::metadata(&host_path) {
        Ok(m) => m,
        Err(e) => {
            results[0] = Value::I32(io_error_to_errno(&e));
            return Ok(());
        }
    };

    // Determine filetype
    let filetype = if metadata.is_dir() {
        FILETYPE_DIRECTORY
    } else if metadata.is_file() {
        FILETYPE_REGULAR_FILE
    } else {
        FILETYPE_UNKNOWN
    };

    // Get timestamps (nanoseconds since epoch)
    let atim = metadata
        .accessed()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mtim = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let ctim = metadata
        .created()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Write filestat_t (64 bytes) to guest memory:
    //   dev:      u64 @ +0
    //   ino:      u64 @ +8
    //   filetype: u8  @ +16
    //   nlink:    u64 @ +24
    //   size:     u64 @ +32
    //   atim:     u64 @ +40
    //   mtim:     u64 @ +48
    //   ctim:     u64 @ +56
    write_u64_le(mem, buf_ptr, 0)?; // dev
    write_u64_le(mem, buf_ptr + 8, 0)?; // ino
    write_u8_le(mem, buf_ptr + 16, filetype)?;
    write_u64_le(mem, buf_ptr + 24, 1)?; // nlink
    write_u64_le(mem, buf_ptr + 32, metadata.len())?;
    write_u64_le(mem, buf_ptr + 40, atim)?;
    write_u64_le(mem, buf_ptr + 48, mtim)?;
    write_u64_le(mem, buf_ptr + 56, ctim)?;

    results[0] = Value::I32(ERRNO_SUCCESS);
    Ok(())
}

pub(crate) fn path_filestat_set_times(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_readlink(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_remove_directory(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_unlink_file(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_rename(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_link(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}

pub(crate) fn path_symlink(
    _caller: &mut Caller,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    results[0] = Value::I32(ERRNO_NOSYS);
    Ok(())
}
