// sf-nano-core build script
//
// Central authority for feature → cfg mapping AND target (arch × os) gating.
// Source code uses `sf_*` cfgs exclusively and must not reference raw
// `cfg(target_os = ...)` / `cfg(target_arch = ...)`; build.rs is the only
// place where those translate into the `sf_*` vocabulary.
//
// Naming scheme:
//   sf_arch_* — selected target architecture (exactly one set when supported)
//   sf_os_*   — selected target OS           (exactly one set when supported)
//   sf_has_*  — derived target capability
//   sf_*      — user-enabled subsystem
//
// Arch cfgs (from CARGO_CFG_TARGET_ARCH):
//   aarch64                → sf_arch_arm64
//   arm + thumbv*-none-*   → sf_arch_thumbm   (arm32 module, Thumb-2 encoding)
//   arm (everything else)  → sf_arch_armv7a   (arm32 module, A32 encoding)
//   x86_64                 → sf_arch_x64
//
// Encoder-variant cfg (independent of sf_arch_*):
//   sf_arm32_isa_thumb — arm32 module emits Thumb-2 via enc_t2.rs. Set for
//     any sf_arch_thumbm target, and also for sf_arch_armv7a when the
//     `thumb2-test` cargo feature is on (used to run the existing armv7-A
//     qemu harness against Thumb-2 output for encoder validation).
//
// OS cfgs (from CARGO_CFG_TARGET_OS):
//   linux   → sf_os_linux   (+ sf_has_posix)
//   macos   → sf_os_macos   (+ sf_has_posix)
//   windows → sf_os_windows
//   none    → sf_os_none    (bare-metal; embedder provides OS shims)
//
// Supported (arch × os) matrix:
//   arm64  × { linux, macos, none }
//   x86_64 × { linux, macos, windows, none }
//   arm    × { linux, none }
//
// Unsupported combos are not validated here — the source simply falls through
// to the emulator backend, matching today's behavior.
//
// Feature → cfg mapping:
//   (derived)      → sf_has_std          (set whenever any feature that needs libstd is on:
//                                          wasi, call-trace, guard-pages)
//   (derived)      → sf_has_guard_pages  (guard-pages + jit + 64-bit + macOS|Linux + x64|arm64)
//   (derived)      → sf_has_debug_regions (set when any consumer of DebugRegion is compiled
//                                          in: sf_ir_dump or sf_jitdump)
//   jit            → sf_jit
//   emulator       → sf_emulator
//   wasi           → sf_wasi_host
//   validator      → sf_module_validator
//   call-trace     → sf_call_trace
//   ir-dump        → sf_ir_dump          (also auto-on when PROFILE=debug; requires std)
//   jitdump        → sf_jitdump          (emits JIT symbol/code info for external profilers
//                                          like samply/perf; requires jit + std)
//
// There is no user-facing `std` feature. libstd availability is derived from
// whichever std-requiring feature the user selected, not requested directly.
//
// All cfgs are declared via `rustc-check-cfg` so typos become compile errors.

use std::env;

const DECLARED_CFGS: &[&str] = &[
    "sf_arch_arm64",
    "sf_arch_armv7a",
    "sf_arch_thumbm",
    "sf_arch_x64",
    "sf_arm32_isa_thumb",
    "sf_os_linux",
    "sf_os_macos",
    "sf_os_windows",
    "sf_os_none",
    "sf_has_posix",
    "sf_has_std",
    "sf_has_guard_pages",
    "sf_has_debug_regions",
    "sf_jit",
    "sf_emulator",
    "sf_wasi_host",
    "sf_module_validator",
    "sf_call_trace",
    "sf_fp_dp",
    "sf_ir_dump",
    "sf_jitdump",
];

fn main() {
    for name in DECLARED_CFGS {
        println!("cargo::rustc-check-cfg=cfg({name})");
    }

    emit_arch_cfgs();
    emit_os_cfgs();
    emit_has_std_cfg();
    emit_subsystem_cfgs();
    // ir_dump and jitdump decisions share inputs with the derived
    // sf_has_debug_regions cfg, so compute them together.
    let want_ir_dump = compute_want_ir_dump();
    let want_jitdump = compute_want_jitdump();
    if want_ir_dump {
        println!("cargo:rustc-cfg=sf_ir_dump");
    }
    if want_jitdump {
        println!("cargo:rustc-cfg=sf_jitdump");
    }
    if want_ir_dump || want_jitdump {
        println!("cargo:rustc-cfg=sf_has_debug_regions");
    }
    emit_guard_pages_cfg();
}

fn emit_arch_cfgs() {
    // Source code uses `sf_arch_*` cfgs instead of raw `target_arch = ...`.
    // Exactly one `sf_arch_*` is set on the three supported architectures;
    // unsupported targets set none and fall through to the emulator.
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    match arch.as_str() {
        "aarch64" => println!("cargo:rustc-cfg=sf_arch_arm64"),
        "arm" => {
            if target.starts_with("thumbv") {
                println!("cargo:rustc-cfg=sf_arch_thumbm");
                println!("cargo:rustc-cfg=sf_arm32_isa_thumb");
                // sf_fp_dp stays off on thumbm: no M-profile FPU offers
                // double-precision arithmetic (FPv5-SP-D16 is SP-only, and
                // no DP variant exists for M-profile). Wasm f64 on thumbm
                // requires a separate legalization story and is out of
                // scope for the initial bring-up.
            } else {
                println!("cargo:rustc-cfg=sf_arch_armv7a");
                // ARMv7-A targets with IDIV always have VFPv3-D16+.
                println!("cargo:rustc-cfg=sf_fp_dp");
                // Opt-in Thumb-2 emit path for encoder validation under
                // the armv7-A qemu harness. A32 Rust ↔ Thumb-2 JIT code
                // bridge via ARM/Thumb interworking at BX/BLX boundaries.
                if env::var_os("CARGO_FEATURE_THUMB2_TEST").is_some() {
                    println!("cargo:rustc-cfg=sf_arm32_isa_thumb");
                }
            }
        }
        "x86_64" => println!("cargo:rustc-cfg=sf_arch_x64"),
        _ => {}
    }
}

fn emit_os_cfgs() {
    // Source code uses `sf_os_*` cfgs instead of raw `target_os = ...`, and
    // `sf_has_posix` as the shorthand for "linux or macos" (shared mmap /
    // sigaction code paths). Windows and bare-metal (`none`) each get their
    // own module; anything else sets no `sf_os_*` at all.
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match os.as_str() {
        "linux" => {
            println!("cargo:rustc-cfg=sf_os_linux");
            println!("cargo:rustc-cfg=sf_has_posix");
        }
        "macos" => {
            println!("cargo:rustc-cfg=sf_os_macos");
            println!("cargo:rustc-cfg=sf_has_posix");
        }
        "windows" => println!("cargo:rustc-cfg=sf_os_windows"),
        "none" => println!("cargo:rustc-cfg=sf_os_none"),
        _ => {}
    }
}

fn emit_has_std_cfg() {
    if has_std_enabled() {
        println!("cargo:rustc-cfg=sf_has_std");
    }
}

fn emit_subsystem_cfgs() {
    if env::var_os("CARGO_FEATURE_JIT").is_some() {
        println!("cargo:rustc-cfg=sf_jit");
    }
    if env::var_os("CARGO_FEATURE_EMULATOR").is_some() {
        println!("cargo:rustc-cfg=sf_emulator");
    }
    if env::var_os("CARGO_FEATURE_WASI").is_some() {
        println!("cargo:rustc-cfg=sf_wasi_host");
    }
    if env::var_os("CARGO_FEATURE_VALIDATOR").is_some() {
        println!("cargo:rustc-cfg=sf_module_validator");
    }
    if env::var_os("CARGO_FEATURE_CALL_TRACE").is_some() {
        println!("cargo:rustc-cfg=sf_call_trace");
    }
}

fn has_std_enabled() -> bool {
    env::var_os("CARGO_FEATURE_WASI").is_some()
        || env::var_os("CARGO_FEATURE_CALL_TRACE").is_some()
        || env::var_os("CARGO_FEATURE_GUARD_PAGES").is_some()
}

fn compute_want_ir_dump() -> bool {
    // ir-dump is a dev-tool feature: always on in debug builds, opt-in for
    // release builds via the `ir-dump` feature.
    //
    // Requires: `jit` (the dumper inspects JIT-only IR/Machine types) AND a
    // feature that pulls in libstd (the dumper writes files). If the build
    // has jit but no std, ir_dump would compile a lot of unused code; we
    // silently drop the cfg in that case.
    //
    // Note: Cargo does not surface `debug_assertions` via CARGO_CFG_*, so we
    // use PROFILE=debug as the proxy for "this is a debug build". Custom
    // profiles inherit from either `dev` (PROFILE=debug) or `release`.
    if env::var_os("CARGO_FEATURE_JIT").is_none() {
        return false;
    }
    if !has_std_enabled() {
        return false;
    }
    let is_debug_profile = env::var("PROFILE").as_deref() == Ok("debug");
    env::var_os("CARGO_FEATURE_IR_DUMP").is_some() || is_debug_profile
}

fn compute_want_jitdump() -> bool {
    // Emits jitdump records so external profilers (samply, perf) can resolve
    // JIT-compiled code regions to symbols. Not a profiler itself — just the
    // exporter side. Requires jit (there's no JIT code without it) and libstd
    // (it writes files). Opt-in via the `jitdump` feature; not auto-enabled
    // in debug builds because it is primarily a release-profile tool.
    if env::var_os("CARGO_FEATURE_JITDUMP").is_none() {
        return false;
    }
    if env::var_os("CARGO_FEATURE_JIT").is_none() {
        return false;
    }
    has_std_enabled()
}

fn emit_guard_pages_cfg() {
    let dominated = env::var_os("CARGO_FEATURE_GUARD_PAGES").is_some()
        && env::var_os("CARGO_FEATURE_JIT").is_some();
    if !dominated {
        return;
    }
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let pw = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    let os_ok = matches!(os.as_str(), "macos" | "linux")
        || (os == "windows" && arch == "x86_64");
    if pw == "64" && os_ok && matches!(arch.as_str(), "x86_64" | "aarch64") {
        println!("cargo:rustc-cfg=sf_has_guard_pages");
    }
}
