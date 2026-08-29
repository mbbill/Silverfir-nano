#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

#[cfg(all(test, not(feature = "memprof")))]
pub(crate) mod test_alloc {
    use core::{
        alloc::{GlobalAlloc, Layout},
        cell::Cell,
    };

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct LayerCensus {
        pub(crate) allocations: usize,
        pub(crate) reallocations: usize,
        pub(crate) allocated_bytes: usize,
        pub(crate) reallocated_bytes: usize,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct Census {
        pub(crate) allocations: usize,
        pub(crate) reallocations: usize,
        pub(crate) allocated_bytes: usize,
        pub(crate) reallocated_bytes: usize,
        pub(crate) function_metadata: LayerCensus,
        pub(crate) test_oracle: LayerCensus,
    }

    impl Census {
        #[cfg(all(sf_interp, not(sf_jit)))]
        pub(crate) fn base(self) -> LayerCensus {
            let classified_allocations = self
                .function_metadata
                .allocations
                .checked_add(self.test_oracle.allocations)
                .expect("classified allocation count overflow");
            let classified_reallocations = self
                .function_metadata
                .reallocations
                .checked_add(self.test_oracle.reallocations)
                .expect("classified reallocation count overflow");
            let classified_allocated_bytes = self
                .function_metadata
                .allocated_bytes
                .checked_add(self.test_oracle.allocated_bytes)
                .expect("classified allocation byte count overflow");
            let classified_reallocated_bytes = self
                .function_metadata
                .reallocated_bytes
                .checked_add(self.test_oracle.reallocated_bytes)
                .expect("classified reallocation byte count overflow");
            LayerCensus {
                allocations: self
                    .allocations
                    .checked_sub(classified_allocations)
                    .expect("classified allocations exceed total"),
                reallocations: self
                    .reallocations
                    .checked_sub(classified_reallocations)
                    .expect("classified reallocations exceed total"),
                allocated_bytes: self
                    .allocated_bytes
                    .checked_sub(classified_allocated_bytes)
                    .expect("classified allocation bytes exceed total"),
                reallocated_bytes: self
                    .reallocated_bytes
                    .checked_sub(classified_reallocated_bytes)
                    .expect("classified reallocation bytes exceed total"),
            }
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) enum AllocationClass {
        #[default]
        Base,
        FunctionMetadata,
        TestOracle,
    }

    std::thread_local! {
        static ENABLED: Cell<bool> = const { Cell::new(false) };
        static CLASS: Cell<AllocationClass> = const { Cell::new(AllocationClass::Base) };
        static COUNTS: Cell<Census> = const { Cell::new(Census {
            allocations: 0,
            reallocations: 0,
            allocated_bytes: 0,
            reallocated_bytes: 0,
            function_metadata: LayerCensus {
                allocations: 0,
                reallocations: 0,
                allocated_bytes: 0,
                reallocated_bytes: 0,
            },
            test_oracle: LayerCensus {
                allocations: 0,
                reallocations: 0,
                allocated_bytes: 0,
                reallocated_bytes: 0,
            },
        }) };
    }

    struct CensusAllocator;

    impl CensusAllocator {
        #[inline]
        fn record_allocation(size: usize) {
            let enabled = ENABLED.try_with(Cell::get).unwrap_or(false);
            if enabled {
                let class = CLASS.try_with(Cell::get).unwrap_or_default();
                let _ = COUNTS.try_with(|counts| {
                    let mut census = counts.get();
                    census.allocations += 1;
                    census.allocated_bytes += size;
                    let layer = match class {
                        AllocationClass::Base => None,
                        AllocationClass::FunctionMetadata => Some(&mut census.function_metadata),
                        AllocationClass::TestOracle => Some(&mut census.test_oracle),
                    };
                    if let Some(layer) = layer {
                        layer.allocations += 1;
                        layer.allocated_bytes += size;
                    }
                    counts.set(census);
                });
            }
        }

        #[inline]
        fn record_reallocation(size: usize) {
            let enabled = ENABLED.try_with(Cell::get).unwrap_or(false);
            if enabled {
                let class = CLASS.try_with(Cell::get).unwrap_or_default();
                let _ = COUNTS.try_with(|counts| {
                    let mut census = counts.get();
                    census.reallocations += 1;
                    census.reallocated_bytes += size;
                    let layer = match class {
                        AllocationClass::Base => None,
                        AllocationClass::FunctionMetadata => Some(&mut census.function_metadata),
                        AllocationClass::TestOracle => Some(&mut census.test_oracle),
                    };
                    if let Some(layer) = layer {
                        layer.reallocations += 1;
                        layer.reallocated_bytes += size;
                    }
                    counts.set(census);
                });
            }
        }
    }

    unsafe impl GlobalAlloc for CensusAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { std::alloc::System.alloc(layout) };
            if !ptr.is_null() {
                Self::record_allocation(layout.size());
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { std::alloc::System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                Self::record_allocation(layout.size());
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let new_ptr = unsafe { std::alloc::System.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                Self::record_reallocation(new_size);
            }
            new_ptr
        }
    }

    #[global_allocator]
    static TEST_ALLOCATOR: CensusAllocator = CensusAllocator;

    pub(crate) fn measure<T>(f: impl FnOnce() -> T) -> (T, Census) {
        COUNTS.with(|counts| counts.set(Census::default()));
        CLASS.with(|class| class.set(AllocationClass::Base));
        ENABLED.with(|enabled| {
            assert!(!enabled.replace(true), "nested allocation census");
        });
        let value = f();
        ENABLED.with(|enabled| enabled.set(false));
        let census = COUNTS.with(Cell::get);
        (value, census)
    }

    pub(crate) struct AllocationClassGuard {
        previous: AllocationClass,
    }

    impl Drop for AllocationClassGuard {
        fn drop(&mut self) {
            CLASS.with(|class| class.set(self.previous));
        }
    }

    pub(crate) fn classify(class: AllocationClass) -> AllocationClassGuard {
        let previous = CLASS.with(|current| current.replace(class));
        AllocationClassGuard { previous }
    }
}

#[cfg(any(sf_has_std, test))]
extern crate std;

pub(crate) mod collections;
// At least one execution engine has to be compiled in; a crate that can parse
// and validate Wasm but not run it is not a useful build. Whether the engine
// you asked for can exist on this ISA is a separate question, answered in
// build.rs (`require_supported_isa`) before any of this compiles.
#[cfg(not(any(sf_jit, sf_interp)))]
compile_error!(
    "sf-nano-core needs at least one execution engine: enable the `jit` feature, \
     the `interp` feature, or both (the default)."
);

pub mod config;
pub mod constants;
pub mod error;
pub mod module;
pub mod op_decoder;
pub mod opcodes;
pub(crate) mod utils;
pub mod value_type;
pub mod vm;

#[cfg(sf_wasi_host)]
pub mod wasi;

// Public re-exports for ergonomic API
pub use config::{Config, ConfigError};
pub use error::WasmError;
pub use module::type_defs::FunctionType;
pub use utils::limits::{Limitable, Limits};
pub use vm::engine::{Engine, Tier};
pub use vm::entities::{Caller, FunctionInst, HostCallback, HostFn};
pub use vm::instance::{
    Func, Import, ImportValue, ImportedFunction, ImportedTableState, ImportedTagState, Instance,
    InstanceInstantiationError, RuntimeWorld,
};
// Each engine publishes one escape hatch for what only it can answer: the
// interpreter's dispatch statistics here, the JIT's native-code question on
// `JitInstanceLease`. They are not counterparts in shape -- the JIT hands
// back a token wrapper and the interpreter lends its body for a closure
// scope -- and they need not be, because each exposes a handful of methods.
// Everything else an embedder needs is on `Instance`.
//
// The predecoded representation (instructions, opcode enum, operand flags)
// stays private: it is how the engine stores a function, not something an
// embedder builds against.
#[cfg(sf_interp)]
pub use vm::interpreter::{FuncRefHost, InterpInstance};
#[cfg(sf_jit)]
pub use vm::jit::arch::active_native_backend_name;
// Compile statistics belong to the JIT *engine*, so they carry its name.
// "Native" is this tree's word for the ISA, which is a different axis --
// see `Engine` versus `active_native_backend_name`.
#[cfg(sf_jit)]
pub use vm::jit::build::{jit_stats_snapshot, JitStatsSnapshot};
#[cfg(sf_jit)]
pub use vm::jit::instantiate::JitInstanceLease;
#[cfg(sf_has_guard_pages)]
use vm::jit::runtime::trap_signal;
pub use vm::link::{InstanceId, LinkRegistry, WorldAccess};
pub use vm::tag::TagIdentity;
pub use vm::value::{RefValue, Value};

#[inline]
pub const fn target_has_simd() -> bool {
    cfg!(sf_has_simd)
}

/// Reset process-global native runtime state that does not track module
/// lifetimes on its own. Harnesses that repeatedly construct and drop native
/// modules can call this between independent runs.
pub fn reset_native_runtime_state() {
    #[cfg(sf_has_guard_pages)]
    {
        trap_signal::reset_debug_state();
        trap_signal::clear_registered_jit_ranges();
    }
}
