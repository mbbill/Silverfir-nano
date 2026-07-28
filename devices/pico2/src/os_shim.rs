//! Bare-metal OS shims for sf-nano-core.
//!
//! sf-nano-core's `runtime/os/none.rs` declares these four symbols
//! `extern "C"`; we resolve them at link time. Backed by a single
//! static arena sized by `config::CODE_ARENA_BYTES` — sf-nano-core's
//! streaming compile path emits directly into the module's persistent
//! buffer (no scratch+swap), so one concurrent allocation is enough.
//!
//! Neither the Cortex-M33 nor the Hazard3 cores on RP2350 have an
//! instruction cache or MPU-enforced W^X by default, so SRAM is always
//! writable and executable. The one thing the finish-write path owes
//! the CPU is a data/instruction barrier (`dsb sy; isb sy` on ARM,
//! `fence rw,rw; fence.i` on RV) so the pipeline discards stale
//! prefetches of newly-emitted instructions. Both are routed through
//! [`crate::arch`].

use core::sync::atomic::{AtomicBool, Ordering};

use crate::config::CODE_ARENA_BYTES;

/// 16-byte-aligned executable arena. This satisfies both ARM and RV32
/// codegen and gives the JIT room to place literal pools without
/// realignment fixups at the arena base.
#[repr(align(16))]
struct CodeArena {
    bytes: [u8; CODE_ARENA_BYTES],
}

static mut CODE_ARENA: CodeArena = CodeArena {
    bytes: [0; CODE_ARENA_BYTES],
};

/// Single-shot flag: `true` while the arena is handed out.
static CODE_ARENA_TAKEN: AtomicBool = AtomicBool::new(false);

/// Reserve the code arena. Returns a pointer to its base if the
/// requested capacity fits and the arena is not already taken;
/// otherwise null (sf-nano-core treats null as "allocation failed").
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_alloc_executable(capacity: usize) -> *mut u8 {
    if capacity > CODE_ARENA_BYTES {
        return core::ptr::null_mut();
    }
    if CODE_ARENA_TAKEN
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return core::ptr::null_mut();
    }
    // SAFETY: `addr_of_mut!` yields a raw pointer without creating a
    // reference to the static mut. CODE_ARENA is 'static so the
    // pointer is valid until process exit (firmware never exits).
    let ptr = core::ptr::addr_of_mut!(CODE_ARENA.bytes).cast::<u8>();
    debug_assert!(ptr as usize % 16 == 0);
    ptr
}

/// Release the arena. sf-nano-core calls this when the owning
/// CodeBuffer is dropped. Subsequent allocs succeed again.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_free_executable(_base: *mut u8, _capacity: usize) {
    CODE_ARENA_TAKEN.store(false, Ordering::Release);
}

/// No-op on RP2350: SRAM is always writable without MPU setup,
/// and we have no W^X page permissions to toggle.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_begin_write_executable(_base: *mut u8, _capacity: usize) {}

/// Make newly-written instructions visible to the CPU's fetch pipeline.
/// `dsb` drains in-flight stores; `isb` flushes the prefetch buffer so
/// stale instructions are discarded. No cache maintenance needed —
/// neither RP2350 core type has an L1 I-cache.
#[unsafe(no_mangle)]
pub extern "C" fn sf_os_finish_write_executable(
    _base: *mut u8,
    _capacity: usize,
    _written_start: usize,
    _written_len: usize,
) {
    crate::arch::dsb();
    crate::arch::isb();
}
