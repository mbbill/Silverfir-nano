//! macOS host primitives for `jitdump`: file open via `fopen`/`fileno`,
//! monotonic time via `mach_absolute_time` + `mach_timebase_info`, and
//! ELF machine arch tag hard-coded to `EM_AARCH64` (jitdump is only
//! interesting on Apple-silicon Macs; the Intel path is unused here).

use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use super::EM_AARCH64;

#[repr(C)]
struct mach_timebase_info {
    numer: u32,
    denom: u32,
}

unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut mach_timebase_info) -> i32;
    fn fopen(
        path: *const core::ffi::c_char,
        mode: *const core::ffi::c_char,
    ) -> *mut core::ffi::c_void;
    fn fileno(stream: *mut core::ffi::c_void) -> i32;
}

pub(super) fn open_tracking_file(path: &Path) -> io::Result<File> {
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

pub(super) fn monotonic_timestamp_nanos() -> u64 {
    unsafe {
        let mut timebase = mach_timebase_info { numer: 0, denom: 0 };
        mach_timebase_info(&mut timebase);
        let ticks = mach_absolute_time();
        ticks.saturating_mul(timebase.numer as u64) / timebase.denom.max(1) as u64
    }
}

pub(super) fn elf_machine_arch() -> u32 {
    EM_AARCH64
}
