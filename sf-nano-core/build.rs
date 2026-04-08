// sf-nano-core build script
//
// Central authority for feature → cfg mapping. Source code uses `sf_*` cfgs;
// Cargo features are only the user-facing knobs. Build.rs validates feature
// combinations against target capabilities and emits cfg flags accordingly.
//
// Naming scheme:
//   sf_has_*  — target capability (auto-derived)
//   sf_*      — user-enabled subsystem
//
// Feature → cfg mapping:
//   (derived)      → sf_has_std          (set whenever any feature that needs libstd is on:
//                                          wasi, call-trace, guard-pages)
//   (derived)      → sf_has_guard_pages  (guard-pages + jit + 64-bit + macOS|Linux + x64|arm64)
//   (derived)      → sf_has_debug_regions (set when any consumer of DebugRegion is compiled
//                                          in: sf_ir_dump or sf_jitdump)
//   jit            → sf_jit
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
    "sf_has_std",
    "sf_has_guard_pages",
    "sf_has_debug_regions",
    "sf_jit",
    "sf_wasi_host",
    "sf_module_validator",
    "sf_call_trace",
    "sf_ir_dump",
    "sf_jitdump",
];

fn main() {
    for name in DECLARED_CFGS {
        println!("cargo::rustc-check-cfg=cfg({name})");
    }

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

fn emit_has_std_cfg() {
    if has_std_enabled() {
        println!("cargo:rustc-cfg=sf_has_std");
    }
}

fn emit_subsystem_cfgs() {
    if env::var_os("CARGO_FEATURE_JIT").is_some() {
        println!("cargo:rustc-cfg=sf_jit");
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
    if pw == "64"
        && matches!(os.as_str(), "macos" | "linux")
        && matches!(arch.as_str(), "x86_64" | "aarch64")
    {
        println!("cargo:rustc-cfg=sf_has_guard_pages");
    }
}
