//! Linux host primitives for `jitdump`: file open via `fopen`/`fileno`,
//! monotonic time via `clock_gettime(CLOCK_MONOTONIC)`, the executable
//! marker mapping `perf inject --jit` pairs a dump with a profile by,
//! and ELF machine arch tag (`EM_AARCH64` on arm64, `EM_X86_64` on
//! x86_64, `EM_RISCV` on RISC-V, else `EM_NONE`).

use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

#[cfg(sf_backend_arm64)]
use super::EM_AARCH64;
#[cfg(not(any(
    sf_backend_arm64,
    sf_backend_riscv64,
    sf_backend_riscv32,
    sf_backend_x64
)))]
use super::EM_NONE;
#[cfg(any(sf_backend_riscv64, sf_backend_riscv32))]
use super::EM_RISCV;
#[cfg(sf_backend_x64)]
use super::EM_X86_64;

const CLOCK_MONOTONIC: i32 = 1;

#[repr(C)]
struct timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

unsafe extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut timespec) -> i32;
    fn fopen(
        path: *const core::ffi::c_char,
        mode: *const core::ffi::c_char,
    ) -> *mut core::ffi::c_void;
    fn fileno(stream: *mut core::ffi::c_void) -> i32;
}

pub(super) fn open_tracking_file(path: &Path) -> io::Result<File> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "jitdump path contains NUL"))?;
    // "w+b", not "wb": the marker mapping below needs a readable fd —
    // mmap(PROT_READ) of an O_WRONLY descriptor fails with EACCES.
    let file_ptr = unsafe { fopen(c_path.as_ptr(), c"w+b".as_ptr()) };
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
        let mut ts = timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        clock_gettime(CLOCK_MONOTONIC, &mut ts);
        (ts.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(ts.tv_nsec as u64)
    }
}

/// Map one executable page of the dump file, which makes the kernel log
/// a `PERF_RECORD_MMAP` carrying the file's path — the marker
/// `perf inject --jit` scans for to pair the dump with the profile.
/// The page is never touched and the mapping intentionally lives for
/// the rest of the process. Best-effort: samply resolves the dump by
/// path and works without the marker.
#[cfg(target_pointer_width = "64")]
pub(super) fn mark_for_perf(file: &File) {
    use std::os::fd::AsRawFd;

    const PROT_READ: i32 = 1;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE: i32 = 2;

    // Signature matches the declaration in vm/jit/runtime/os/posix.rs;
    // the two must agree (clashing_extern_declarations).
    unsafe extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    }

    // Length 1 maps a single page. The header was already written, so
    // the range is backed; nothing ever reads through this mapping.
    let mapped = unsafe {
        mmap(
            core::ptr::null_mut(),
            1,
            PROT_READ | PROT_EXEC,
            MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    if mapped as isize == -1 {
        // SF_JITDUMP is an explicit request to feed perf; a silent
        // marker failure leaves every profile unsymbolized with no clue.
        std::eprintln!(
            "[jitdump] marker mapping failed; perf inject --jit will not resolve symbols"
        );
    }
}

/// The 32-bit Linux targets run under qemu-user in CI, which perf does
/// not profile, and `off_t`'s width there depends on the libc — the
/// marker is not worth a per-libc mmap ABI. samply still works.
#[cfg(not(target_pointer_width = "64"))]
pub(super) fn mark_for_perf(_file: &File) {}

pub(super) fn elf_machine_arch() -> u32 {
    #[cfg(sf_backend_arm64)]
    {
        EM_AARCH64
    }
    #[cfg(any(sf_backend_riscv64, sf_backend_riscv32))]
    {
        EM_RISCV
    }
    #[cfg(sf_backend_x64)]
    {
        EM_X86_64
    }
    #[cfg(not(any(
        sf_backend_arm64,
        sf_backend_riscv64,
        sf_backend_riscv32,
        sf_backend_x64
    )))]
    {
        EM_NONE
    }
}
