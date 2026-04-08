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
//   (derived)      → sf_has_std        (CARGO_FEATURE_STD, pulled in by wasi/call-trace/guard-pages)
//   (derived)      → sf_has_guard_pages (guard-pages + jit + 64-bit + macOS|Linux + x64|arm64)
//   jit            → sf_jit
//   wasi           → sf_wasi_host
//   validator      → sf_module_validator
//   call-trace     → sf_call_trace
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
];

fn main() {
    for name in DECLARED_CFGS {
        println!("cargo::rustc-check-cfg=cfg({name})");
    }

    emit_has_std_cfg();
    emit_subsystem_cfgs();
    emit_guard_pages_cfg();
}

fn emit_has_std_cfg() {
    if env::var_os("CARGO_FEATURE_STD").is_some() {
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
        if env::var_os("CARGO_FEATURE_STD").is_none() {
            panic!(
                "sf-nano-core: feature `call-trace` requires libstd, \
                 but the `std` feature is not enabled"
            );
        }
        println!("cargo:rustc-cfg=sf_call_trace");
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
