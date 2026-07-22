//! Runtime configuration for sf-nano-core.
//!
//! A one-shot global installed by the embedder via [`set_runtime_config`].
//! Hosted targets get defaults suitable for native JIT workloads, while the
//! bare-metal target (`sf_os_none`) has an explicit zero default — the embedder
//! **must** call [`set_runtime_config`] before any `Instance::new()` /
//! `CodeBuffer::new()` path, otherwise those fail with a clear error.
//!
//! Design: see `docs/RUNTIME_CONFIG_AND_OS_MEMORY.md`.
//!
//! # Threading
//!
//! The configuration is designed for single-threaded startup: the
//! embedder calls `set_runtime_config` once, before any other
//! sf-nano-core API is used. No concurrent configure-and-use scenario
//! is supported. Readers acquire-load a state byte to get a memory
//! fence against a concurrent writer, but we do not branch on the
//! state — the underlying storage holds a valid `RuntimeConfig` at
//! every point (default → user value after init).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU8, Ordering};

/// Runtime-wide tunable sizes and limits.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    /// Module-wide executable-code arena size. This is the hard limit for
    /// native code emitted for one module and the number of bytes requested
    /// from the executable-memory allocator when a `CodeBuffer` is created.
    pub code_arena_bytes: usize,

    /// Maximum 64-KiB Wasm pages a single linear memory is allowed to
    /// reach. Enforced at instantiation for the initial page count and
    /// at `memory.grow` for the requested new size. In JIT builds,
    /// local memory backing is materialized after compilation but this
    /// quota is still checked before compilation starts.
    /// `u32` is enough for wasm32 (ceiling 65536). Memory64 modules
    /// are allowed but cannot exceed `u32::MAX` pages through this
    /// config; practical MCU targets set this to 1–16 anyway.
    pub wasm_memory_max_pages: u32,

    /// Bytes reserved for the Wasm operand/call stack that backs every
    /// JIT invoke. The eval path allocates one of these at the start
    /// of each `invoke` call. Must be a multiple of 8 (the stack is
    /// indexed in u64 slots). Hosted default is 2 MiB; MCU targets
    /// should set this to single-digit KiB for simple modules.
    pub wasm_stack_bytes: usize,

    /// Per-function compiler-time RAM budget in bytes. Hosted builds
    /// default this to `u32::MAX`, which preserves the existing full
    /// optimization path. Embedded targets use this to keep oversized
    /// functions out of the full compiler pipeline.
    pub compiler_ram_budget_bytes: u32,
}

impl RuntimeConfig {
    /// Compile-time default. Hosted 64-bit targets reserve enough executable
    /// address space for large native JIT workloads such as FFmpeg. Hosted
    /// 32-bit targets retain the smaller arena to avoid exhausting their
    /// limited address space. `sf_os_none` gets zeros that
    /// force the embedder to override (see `ConfigError::NotInitialized`
    /// on the consumer side).
    #[cfg(not(sf_os_none))]
    pub const DEFAULT: Self = {
        #[cfg(target_pointer_width = "64")]
        const CODE_DEFAULT: usize = 64 * 1024 * 1024;
        #[cfg(not(target_pointer_width = "64"))]
        const CODE_DEFAULT: usize = 16 * 1024 * 1024;
        Self {
            code_arena_bytes: CODE_DEFAULT,
            wasm_memory_max_pages: 65536,
            // 2 MiB matches the former `constants::MAX_STACK_SIZE`.
            wasm_stack_bytes: 2 * 1024 * 1024,
            compiler_ram_budget_bytes: u32::MAX,
        }
    };

    /// Bare-metal default: zeros. The embedder MUST override via
    /// [`set_runtime_config`]. Any `Instance::new()` / `CodeBuffer::new()`
    /// path before that override fails cleanly.
    #[cfg(sf_os_none)]
    pub const DEFAULT: Self = Self {
        code_arena_bytes: 0,
        wasm_memory_max_pages: 0,
        wasm_stack_bytes: 0,
        compiler_ram_budget_bytes: 0,
    };
}

/// Error returned by [`set_runtime_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    /// [`set_runtime_config`] has already been called successfully.
    /// sf-nano-core's configuration is write-once.
    AlreadyInitialized,
}

// --- Internal state ---------------------------------------------------

const STATE_UNINIT: u8 = 0;
const STATE_WRITING: u8 = 1;
const STATE_READY: u8 = 2;

static STATE: AtomicU8 = AtomicU8::new(STATE_UNINIT);

struct ConfigStorage(UnsafeCell<RuntimeConfig>);
// SAFETY: all mutation is serialized through the STATE CAS. Readers
// only observe either the compile-time default (STATE_UNINIT) or a
// fully-written embedder value (STATE_READY). Cross-thread access is
// not supported — see the module-level threading note.
unsafe impl Sync for ConfigStorage {}

static STORAGE: ConfigStorage = ConfigStorage(UnsafeCell::new(RuntimeConfig::DEFAULT));

/// Install the runtime configuration. Must be called at most once,
/// before any Store / CodeBuffer / MemInst is created. Returns
/// `Err(AlreadyInitialized)` on a second call.
pub fn set_runtime_config(cfg: RuntimeConfig) -> Result<(), ConfigError> {
    STATE
        .compare_exchange(
            STATE_UNINIT,
            STATE_WRITING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .map_err(|_| ConfigError::AlreadyInitialized)?;

    // SAFETY: CAS above serialized us. No other writer can reach this
    // point until STATE transitions back to something != WRITING,
    // which only happens via our store below. Readers of STORAGE
    // either saw STATE_UNINIT (and read the compile-time default from
    // the same slot, which is valid) or will see STATE_READY after
    // our release-store and the fully-written value.
    unsafe {
        *STORAGE.0.get() = cfg;
    }

    STATE.store(STATE_READY, Ordering::Release);
    Ok(())
}

/// Read the active runtime configuration. Returns the compile-time
/// default if [`set_runtime_config`] has not yet been called.
#[inline]
pub fn runtime_config() -> &'static RuntimeConfig {
    // Acquire fence pairs with the Release store in set_runtime_config.
    // We don't branch on the value — the storage slot is always a
    // valid RuntimeConfig (either DEFAULT or the embedder's override).
    let _ = STATE.load(Ordering::Acquire);
    // SAFETY: see the Sync impl and module docs. Under the single-
    // threaded-init contract, no torn reads are possible.
    unsafe { &*STORAGE.0.get() }
}

/// True once an embedder has successfully installed a config via
/// [`set_runtime_config`]. Useful for bare-metal call sites that
/// want to give a specific "embedder did not configure" error
/// instead of proceeding with the zero default.
#[inline]
pub fn runtime_config_is_initialized() -> bool {
    STATE.load(Ordering::Acquire) == STATE_READY
}

#[cfg(test)]
mod tests {
    use super::RuntimeConfig;

    #[cfg(all(not(sf_os_none), target_pointer_width = "64"))]
    #[test]
    fn hosted_64_bit_default_supports_large_jit_modules() {
        assert_eq!(RuntimeConfig::DEFAULT.code_arena_bytes, 64 * 1024 * 1024);
    }

    #[cfg(all(not(sf_os_none), target_pointer_width = "32"))]
    #[test]
    fn hosted_32_bit_default_preserves_address_space() {
        assert_eq!(RuntimeConfig::DEFAULT.code_arena_bytes, 16 * 1024 * 1024);
    }
}
