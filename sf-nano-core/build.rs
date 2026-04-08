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
//   (derived)      → sf_has_std        (set whenever any feature that needs libstd is on:
//                                        wasi, call-trace, guard-pages)
//   (derived)      → sf_has_guard_pages (guard-pages + jit + 64-bit + macOS|Linux + x64|arm64)
//   jit            → sf_jit
//   wasi           → sf_wasi_host
//   validator      → sf_module_validator
//   call-trace     → sf_call_trace
//   ir-dump        → sf_ir_dump        (also auto-on when PROFILE=debug)
//
// There is no user-facing `std` feature. libstd availability is derived from
// whichever std-requiring feature the user selected, not requested directly.
//
// All cfgs are declared via `rustc-check-cfg` so typos become compile errors.

use std::env;

const DECLARED_CFGS: &[&str] = &[
    "sf_has_std",
    "sf_has_guard_pages",
    "sf_jit",
    "sf_wasi_host",
    "sf_module_validator",
    "sf_call_trace",
    "sf_ir_dump",
];

fn main() {
    for name in DECLARED_CFGS {
        println!("cargo::rustc-check-cfg=cfg({name})");
    }

    emit_has_std_cfg();
    emit_subsystem_cfgs();
    emit_ir_dump_cfg();
    emit_guard_pages_cfg();
}

fn emit_has_std_cfg() {
    let needs_std = env::var_os("CARGO_FEATURE_WASI").is_some()
        || env::var_os("CARGO_FEATURE_CALL_TRACE").is_some()
        || env::var_os("CARGO_FEATURE_GUARD_PAGES").is_some();
    if needs_std {
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

fn emit_ir_dump_cfg() {
    // ir-dump is a dev-tool feature: always on in debug builds, opt-in for
    // release builds via the `ir-dump` feature. Only meaningful when `jit` is
    // also enabled (the dumper inspects JIT-only IR/Machine types).
    //
    // Note: Cargo does not surface `debug_assertions` via CARGO_CFG_*, so we
    // use PROFILE=debug as the proxy for "this is a debug build". Custom
    // profiles inherit from either `dev` (PROFILE=debug) or `release`.
    if env::var_os("CARGO_FEATURE_JIT").is_none() {
        return;
    }
    let is_debug_profile = env::var("PROFILE").as_deref() == Ok("debug");
    let wanted = env::var_os("CARGO_FEATURE_IR_DUMP").is_some() || is_debug_profile;
    if wanted {
        println!("cargo:rustc-cfg=sf_ir_dump");
    }
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
