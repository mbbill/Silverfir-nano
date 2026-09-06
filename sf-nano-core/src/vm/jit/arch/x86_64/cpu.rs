//! Host CPU feature probe for the x86_64 backend.
//!
//! SIMD lowering emits SSSE3 (`pshufb`) and SSE4.1 (`ptest`, `pmovsx*`,
//! `pmovzx*`, `pmulld`, `roundps`/`roundpd`). Those instructions run on the
//! host at execution time; they are never produced by rustc for this crate,
//! which uses no x86 intrinsics. The requirement is therefore a property of
//! the running CPU, and is checked here rather than asserted in `build.rs`
//! against the build target's `target_feature` baseline.
//!
//! Scalar lowering retains an x86_64 SSE2 fallback, so a CPU without
//! SSSE3/SSE4.1 still compiles and runs non-SIMD modules; only modules that
//! reach SIMD lowering are rejected.

use core::arch::x86_64::{__cpuid, __cpuid_count};
use core::sync::atomic::{AtomicU8, Ordering};

use crate::error::WasmError;

/// CPUID leaf 1 feature bits for the instruction sets SIMD lowering emits.
const LEAF1_ECX_SSSE3: u32 = 1 << 9;
const LEAF1_ECX_SSE4_1: u32 = 1 << 19;
const LEAF1_EDX_SSE2: u32 = 1 << 26;

const PROBE_PENDING: u8 = 0;
const PROBE_SUPPORTED: u8 = 1;
const PROBE_UNSUPPORTED: u8 = 2;

/// Cached result of `probe()`. Racing threads may probe concurrently; CPUID is
/// pure, so every racer computes the same value and the store is idempotent.
static SIMD_SUPPORT: AtomicU8 = AtomicU8::new(PROBE_PENDING);
static BEXTR_PREFERENCE: AtomicU8 = AtomicU8::new(PROBE_PENDING);

/// AMD's single-operation BEXTR can shorten a shift-and-mask dependency.
/// Intel implementations commonly use two operations, so retain their
/// existing lowering. This is a CPU cost preference, not an ISA requirement.
/// See Agner Fog's instruction tables, Zen 3 and Skylake BEXTR entries.
pub(super) fn prefer_bextr() -> bool {
    let mut state = BEXTR_PREFERENCE.load(Ordering::Relaxed);
    if state == PROBE_PENDING {
        let vendor = __cpuid(0);
        let amd = vendor.ebx == u32::from_le_bytes(*b"Auth")
            && vendor.edx == u32::from_le_bytes(*b"enti")
            && vendor.ecx == u32::from_le_bytes(*b"cAMD");
        let supported = amd && vendor.eax >= 7 && __cpuid_count(7, 0).ebx & (1 << 3) != 0;
        state = if supported {
            PROBE_SUPPORTED
        } else {
            PROBE_UNSUPPORTED
        };
        BEXTR_PREFERENCE.store(state, Ordering::Relaxed);
    }
    state == PROBE_SUPPORTED
}

fn probe() -> u8 {
    // Leaf 1 is architecturally present on every x86_64 CPU, which is why
    // `__cpuid` is a safe call for this target.
    let leaf1 = __cpuid(1);
    let supported = leaf1.edx & LEAF1_EDX_SSE2 != 0
        && leaf1.ecx & LEAF1_ECX_SSSE3 != 0
        && leaf1.ecx & LEAF1_ECX_SSE4_1 != 0;
    if supported {
        PROBE_SUPPORTED
    } else {
        PROBE_UNSUPPORTED
    }
}

/// Reject SIMD lowering when the host CPU cannot execute the instructions it
/// emits. Called from the SIMD lowering entry points, so the cost is one
/// relaxed load per lowered SIMD instruction at compile time.
#[inline]
pub(super) fn require_simd_support() -> Result<(), WasmError> {
    let mut state = SIMD_SUPPORT.load(Ordering::Relaxed);
    if state == PROBE_PENDING {
        state = probe();
        SIMD_SUPPORT.store(state, Ordering::Relaxed);
    }
    if state == PROBE_SUPPORTED {
        Ok(())
    } else {
        Err(WasmError::invalid("SIMD is not supported on this CPU"))
    }
}
