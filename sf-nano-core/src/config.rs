//! What an engine is configured with.
//!
//! A [`Config`] is a plain value the embedder builds and hands to
//! [`crate::Engine`]; the engine copies it into every instance it creates.
//! Nothing here is process-wide, so two engines in one process may run
//! different tiers with different budgets, and configuring one never
//! affects the other.
//!
//! Hosted targets get defaults suited to native JIT workloads. The
//! bare-metal target (`sf_os_none`) has none: its defaults are zero and
//! [`crate::Engine::new`] rejects them, because a sensible number for one
//! board is a fatal one for the next. That rejection happens once, at
//! engine construction, before any module is touched.
//!
//! Design: see `docs/RUNTIME_CONFIG_AND_OS_MEMORY.md`.

use crate::vm::engine::Tier;

/// Tunable sizes and limits for one engine.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    tier: Tier,
    code_arena_bytes: usize,
    wasm_memory_max_pages: u32,
    wasm_stack_bytes: usize,
    compiler_ram_budget_bytes: u32,
    parallel_compilation: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// This target's defaults, on the default tier.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tier: Tier::DEFAULT,
            code_arena_bytes: DEFAULT_CODE_ARENA_BYTES,
            wasm_memory_max_pages: DEFAULT_WASM_MEMORY_MAX_PAGES,
            wasm_stack_bytes: DEFAULT_WASM_STACK_BYTES,
            compiler_ram_budget_bytes: DEFAULT_COMPILER_RAM_BUDGET_BYTES,
            parallel_compilation: DEFAULT_PARALLEL_COMPILATION,
        }
    }

    /// Which engine runs modules instantiated in this configuration.
    #[must_use]
    pub const fn tier(mut self, tier: Tier) -> Self {
        self.tier = tier;
        self
    }

    /// Module-wide executable-code arena size: the hard limit on native
    /// code emitted for one module, and the bytes requested from the
    /// executable-memory allocator when a code buffer is created.
    #[must_use]
    pub const fn code_arena_bytes(mut self, bytes: usize) -> Self {
        self.code_arena_bytes = bytes;
        self
    }

    /// Maximum 64-KiB Wasm pages a single linear memory may reach.
    ///
    /// Enforced at instantiation for the initial page count and at
    /// `memory.grow` for the requested size. Memory64 modules are allowed
    /// but cannot exceed `u32::MAX` pages through this.
    #[must_use]
    pub const fn wasm_memory_max_pages(mut self, pages: u32) -> Self {
        self.wasm_memory_max_pages = pages;
        self
    }

    /// Bytes for the Wasm operand/call stack backing an invocation.
    /// Indexed in `u64` slots. Both engines size their stack from this.
    #[must_use]
    pub const fn wasm_stack_bytes(mut self, bytes: usize) -> Self {
        self.wasm_stack_bytes = bytes;
        self
    }

    /// Per-function compile-time RAM budget. A function estimated above it
    /// takes the template path instead of the full pipeline.
    #[must_use]
    pub const fn compiler_ram_budget_bytes(mut self, bytes: u32) -> Self {
        self.compiler_ram_budget_bytes = bytes;
        self
    }

    /// Whether hosted eager compilation may spread a module's functions
    /// across worker threads. Turning it off selects the serial
    /// implementation of the same pipeline, which is what a compiler
    /// benchmark wants so its numbers do not depend on core count.
    #[must_use]
    pub const fn parallel_compilation(mut self, on: bool) -> Self {
        self.parallel_compilation = on;
        self
    }

    // --- readers ---

    #[inline]
    pub const fn get_tier(&self) -> Tier {
        self.tier
    }
    #[inline]
    pub const fn get_code_arena_bytes(&self) -> usize {
        self.code_arena_bytes
    }
    #[inline]
    pub const fn get_wasm_memory_max_pages(&self) -> u32 {
        self.wasm_memory_max_pages
    }
    #[inline]
    pub const fn get_wasm_stack_bytes(&self) -> usize {
        self.wasm_stack_bytes
    }
    #[inline]
    pub const fn get_compiler_ram_budget_bytes(&self) -> u32 {
        self.compiler_ram_budget_bytes
    }
    #[inline]
    pub const fn get_parallel_compilation(&self) -> bool {
        self.parallel_compilation
    }

    /// The budgets that must be set to run anything, checked once by
    /// [`crate::Engine::new`] so a bare-metal embedder hears about a
    /// missing number before a module rather than during one.
    pub(crate) const fn validate(&self) -> Result<(), ConfigError> {
        if self.wasm_memory_max_pages == 0 {
            return Err(ConfigError::Unconfigured("wasm_memory_max_pages"));
        }
        if self.wasm_stack_bytes == 0 {
            return Err(ConfigError::Unconfigured("wasm_stack_bytes"));
        }
        // The code arena is the JIT's. An interpreter-only engine has no
        // use for one and should not be made to invent a number.
        #[cfg(sf_jit)]
        if self.tier.is_jit() && self.code_arena_bytes == 0 {
            return Err(ConfigError::Unconfigured("code_arena_bytes"));
        }
        Ok(())
    }
}

/// Why a [`Config`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    /// A budget with no usable default on this target was left at zero.
    /// The payload names the field.
    Unconfigured(&'static str),
}

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unconfigured(field) => write!(
                f,
                "Config::{field} has no default on this target and must be set"
            ),
        }
    }
}

// --- Per-target defaults ----------------------------------------------

/// Hosted 64-bit reserves enough executable address space for large
/// workloads; hosted 32-bit keeps a smaller arena so it does not exhaust
/// its address space.
#[cfg(all(not(sf_os_none), target_pointer_width = "64"))]
const DEFAULT_CODE_ARENA_BYTES: usize = 64 * 1024 * 1024;
#[cfg(all(not(sf_os_none), not(target_pointer_width = "64")))]
const DEFAULT_CODE_ARENA_BYTES: usize = 16 * 1024 * 1024;
#[cfg(not(sf_os_none))]
const DEFAULT_WASM_MEMORY_MAX_PAGES: u32 = 65536;
/// 2 MiB, matching the former `constants::MAX_STACK_SIZE`.
#[cfg(not(sf_os_none))]
const DEFAULT_WASM_STACK_BYTES: usize = 2 * 1024 * 1024;
#[cfg(not(sf_os_none))]
const DEFAULT_COMPILER_RAM_BUDGET_BYTES: u32 = u32::MAX;
#[cfg(not(sf_os_none))]
const DEFAULT_PARALLEL_COMPILATION: bool = true;

// Bare metal: zeros, so `Engine::new` refuses rather than inventing a
// budget for a board it knows nothing about.
#[cfg(sf_os_none)]
const DEFAULT_CODE_ARENA_BYTES: usize = 0;
#[cfg(sf_os_none)]
const DEFAULT_WASM_MEMORY_MAX_PAGES: u32 = 0;
#[cfg(sf_os_none)]
const DEFAULT_WASM_STACK_BYTES: usize = 0;
#[cfg(sf_os_none)]
const DEFAULT_COMPILER_RAM_BUDGET_BYTES: u32 = 0;
#[cfg(sf_os_none)]
const DEFAULT_PARALLEL_COMPILATION: bool = false;

#[cfg(test)]
mod tests {
    use super::{Config, ConfigError};

    #[test]
    fn builder_is_chainable_and_readable() {
        let cfg = Config::new()
            .wasm_stack_bytes(4096)
            .parallel_compilation(false);
        assert_eq!(cfg.get_wasm_stack_bytes(), 4096);
        assert!(!cfg.get_parallel_compilation());
    }

    #[test]
    fn a_zeroed_budget_is_rejected_by_name() {
        let cfg = Config::new().wasm_stack_bytes(0);
        assert_eq!(
            cfg.validate(),
            Err(ConfigError::Unconfigured("wasm_stack_bytes"))
        );
    }

    /// The whole point of the change: two configurations coexist, because
    /// neither is a process-wide setting.
    #[test]
    fn configs_are_independent_values() {
        let a = Config::new().wasm_stack_bytes(1024);
        let b = Config::new().wasm_stack_bytes(2048);
        assert_eq!(a.get_wasm_stack_bytes(), 1024);
        assert_eq!(b.get_wasm_stack_bytes(), 2048);
    }
}
