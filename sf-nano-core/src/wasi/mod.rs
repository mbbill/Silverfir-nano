//! WASI preview1 support for sf-nano-core.
//!
//! Gated behind `#[cfg(feature = "wasi")]`. Uses a thread-local `WasiCtx`
//! so that plain `fn`-pointer `ExternalFn` callbacks can access WASI state
//! without closures.

use std::collections::{HashMap, HashSet};
use std::format;
use std::path::PathBuf;
use std::string::{String, ToString};
use std::thread_local;
use std::vec;
use std::vec::Vec;

use crate::error::WasmError;
use crate::vm::entities::{Caller, ExternalFn};
use crate::vm::instance::Import;
use crate::vm::value::Value;

pub(crate) mod preview1;

pub const WASI_SNAPSHOT_PREVIEW1: &str = "wasi_snapshot_preview1";
pub const WASI_UNSTABLE: &str = "wasi_unstable";

// ---------------------------------------------------------------------------
// WasiCtx — runtime state for a WASI instance
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PreopenDir {
    pub guest_path: String,
    pub host_path: PathBuf,
}

pub enum FdEntry {
    Dir {
        host_path: PathBuf,
        rights_base: u64,
        rights_inh: u64,
    },
    File {
        file: std::fs::File,
        host_path: PathBuf,
        rights_base: u64,
        rights_inh: u64,
        fdflags: u16,
    },
}

// FdEntry contains File which is not Debug, provide manual impl
impl std::fmt::Debug for FdEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FdEntry::Dir { host_path, .. } => {
                f.debug_struct("Dir").field("host_path", host_path).finish()
            }
            FdEntry::File { host_path, .. } => f
                .debug_struct("File")
                .field("host_path", host_path)
                .finish(),
        }
    }
}

pub struct WasiCtx {
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub preopens: Vec<PreopenDir>,
    pub next_fd: i32,
    pub fds: HashMap<i32, FdEntry>,
    pub closed_preopens: HashSet<i32>,
    pub closed_stdio: HashSet<i32>,
}

impl WasiCtx {
    pub fn new(args: Vec<String>, env: Vec<(String, String)>, preopens: Vec<PreopenDir>) -> Self {
        let next_fd = 3 + preopens.len() as i32;
        Self {
            args,
            env,
            preopens,
            next_fd,
            fds: HashMap::new(),
            closed_preopens: HashSet::new(),
            closed_stdio: HashSet::new(),
        }
    }

    pub fn alloc_fd(&mut self, entry: FdEntry) -> i32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        self.fds.insert(fd, entry);
        fd
    }
}

// ---------------------------------------------------------------------------
// WasiContextBuilder
// ---------------------------------------------------------------------------

pub struct WasiContextBuilder {
    args: Vec<String>,
    env: Vec<(String, String)>,
    preopens: Vec<PreopenDir>,
}

impl WasiContextBuilder {
    pub fn new() -> Self {
        Self {
            args: vec!["program".into()],
            env: vec![],
            preopens: vec![],
        }
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.args = args.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    pub fn env<K: AsRef<str>, V: AsRef<str>>(mut self, k: K, v: V) -> Self {
        self.env
            .push((k.as_ref().to_string(), v.as_ref().to_string()));
        self
    }

    pub fn inherit_env(mut self) -> Self {
        for (k, v) in std::env::vars() {
            self.env.push((k, v));
        }
        self
    }

    pub fn preopen_dir<P: Into<PathBuf>, S: AsRef<str>>(mut self, guest: S, host: P) -> Self {
        self.preopens.push(PreopenDir {
            guest_path: guest.as_ref().to_string(),
            host_path: host.into(),
        });
        self
    }

    pub fn build(self) -> WasiCtx {
        WasiCtx::new(self.args, self.env, self.preopens)
    }
}

impl Default for WasiContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Thread-local WasiCtx for fn-pointer ExternalFn callbacks
// ---------------------------------------------------------------------------

use std::cell::RefCell;

thread_local! {
    static WASI_CTX: RefCell<Option<WasiCtx>> = const { RefCell::new(None) };
}

/// Install a WasiCtx as the active context for the current thread.
/// Must be called before invoking any WASM function that uses WASI imports.
pub fn set_wasi_ctx(ctx: WasiCtx) {
    WASI_CTX.with(|cell| {
        *cell.borrow_mut() = Some(ctx);
    });
}

/// Remove and return the active WASI context (e.g., after execution finishes).
pub fn take_wasi_ctx() -> Option<WasiCtx> {
    WASI_CTX.with(|cell| cell.borrow_mut().take())
}

/// Access the thread-local WasiCtx. Panics if not installed.
fn with_ctx<R>(f: impl FnOnce(&WasiCtx) -> R) -> R {
    WASI_CTX.with(|cell| {
        let borrow = cell.borrow();
        let ctx = borrow
            .as_ref()
            .expect("WASI context not set; call set_wasi_ctx() before execution");
        f(ctx)
    })
}

/// Access the thread-local WasiCtx mutably. Panics if not installed.
fn with_ctx_mut<R>(f: impl FnOnce(&mut WasiCtx) -> R) -> R {
    WASI_CTX.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let ctx = borrow
            .as_mut()
            .expect("WASI context not set; call set_wasi_ctx() before execution");
        f(ctx)
    })
}

// ---------------------------------------------------------------------------
// WASI Import generation
// ---------------------------------------------------------------------------

/// Generate WASI imports for a given module namespace.
fn wasi_imports_for(module: &str) -> Vec<Import> {
    macro_rules! wasi {
        ($name:literal, $f:expr) => {
            Import::func(module, $name, $f)
        };
    }

    vec![
        wasi!("args_sizes_get", preview1::args_sizes_get),
        wasi!("args_get", preview1::args_get),
        wasi!("environ_sizes_get", preview1::environ_sizes_get),
        wasi!("environ_get", preview1::environ_get),
        wasi!("fd_write", preview1::fd_write),
        wasi!("fd_read", preview1::fd_read),
        wasi!("fd_close", preview1::fd_close),
        wasi!("fd_seek", preview1::fd_seek),
        wasi!("fd_tell", preview1::fd_tell),
        wasi!("fd_fdstat_get", preview1::fd_fdstat_get),
        wasi!("fd_fdstat_set_flags", preview1::fd_fdstat_set_flags),
        wasi!("fd_fdstat_set_rights", preview1::fd_fdstat_set_rights),
        wasi!("fd_prestat_get", preview1::fd_prestat_get),
        wasi!("fd_prestat_dir_name", preview1::fd_prestat_dir_name),
        wasi!("fd_filestat_get", preview1::fd_filestat_get),
        wasi!("fd_filestat_set_size", preview1::fd_filestat_set_size),
        wasi!("fd_filestat_set_times", preview1::fd_filestat_set_times),
        wasi!("fd_sync", preview1::fd_sync),
        wasi!("fd_datasync", preview1::fd_datasync),
        wasi!("fd_renumber", preview1::fd_renumber),
        wasi!("fd_readdir", preview1::fd_readdir),
        wasi!("fd_pread", preview1::fd_pread),
        wasi!("fd_pwrite", preview1::fd_pwrite),
        wasi!("fd_allocate", preview1::fd_allocate),
        wasi!("fd_advise", preview1::fd_advise),
        wasi!("clock_time_get", preview1::clock_time_get),
        wasi!("clock_res_get", preview1::clock_res_get),
        wasi!("random_get", preview1::random_get),
        wasi!("proc_exit", preview1::proc_exit),
        wasi!("sched_yield", preview1::sched_yield),
        wasi!("sock_shutdown", preview1::sock_shutdown),
        wasi!("poll_oneoff", preview1::poll_oneoff),
        wasi!("path_create_directory", preview1::path_create_directory),
        wasi!("path_filestat_get", preview1::path_filestat_get),
        wasi!("path_filestat_set_times", preview1::path_filestat_set_times),
        wasi!("path_open", preview1::path_open),
        wasi!("path_readlink", preview1::path_readlink),
        wasi!("path_remove_directory", preview1::path_remove_directory),
        wasi!("path_unlink_file", preview1::path_unlink_file),
        wasi!("path_rename", preview1::path_rename),
        wasi!("path_link", preview1::path_link),
        wasi!("path_symlink", preview1::path_symlink),
    ]
}

/// Generate all WASI imports (preview1 + unstable) for use with `Instance::new()`.
pub fn wasi_imports() -> Vec<Import> {
    let mut imports = wasi_imports_for(WASI_SNAPSHOT_PREVIEW1);
    imports.extend(wasi_imports_for(WASI_UNSTABLE));
    imports
}
