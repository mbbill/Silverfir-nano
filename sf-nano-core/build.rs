// sf-nano-core build script
//
// Central authority for feature → cfg mapping AND target (arch × os) gating.
// Source code uses `sf_*` cfgs exclusively and must not reference raw
// `cfg(target_os = ...)` / `cfg(target_arch = ...)`; build.rs is the only
// place where those translate into the `sf_*` vocabulary.
//
// Naming scheme:
//   sf_backend_* — selected execution backend ABI/codegen target
//                  (exactly one set when supported)
//   sf_os_*      — selected target OS           (exactly one set when supported)
//   sf_has_*     — derived target capability
//   sf_*         — user-enabled subsystem
//
// Backend cfgs:
//   aarch64                → sf_backend_arm64
//   arm + thumbv*-none-*   → sf_backend_thumbm   (arm32 module, Thumb-2 encoding)
//   arm (everything else)  → sf_backend_armv7a   (arm32 module, A32 encoding)
//   x86_64                 → sf_backend_x64
//   feature backend-emu64  → sf_backend_emu64
//   feature backend-emu32  → sf_backend_emu32
//
// Encoder-variant cfg (independent of sf_backend_*):
//   sf_arm32_isa_thumb — arm32 module emits Thumb-2 via enc_t2.rs. Set for
//     any sf_backend_thumbm target, and also for sf_backend_armv7a when the
//     `thumb2-test` cargo feature is on (used to run the existing armv7-A
//     qemu harness against Thumb-2 output for encoder validation).
//
// OS cfgs (from CARGO_CFG_TARGET_OS):
//   linux   → sf_os_linux   (+ sf_has_posix)
//   macos   → sf_os_macos   (+ sf_has_posix)
//   windows → sf_os_windows
//   none    → sf_os_none    (bare-metal; embedder provides OS shims)
//
// Supported (backend × os) matrix:
//   arm64  × { linux, macos, none }
//   x86_64 × { linux, macos, windows, none }
//   arm    × { linux, none }
//   emu64  × { linux, macos, windows, none, ...host OS passthrough... }
//   emu32  × { linux, macos, windows, none, ...host OS passthrough... }
//
// Unsupported combos are not validated here. If no host-native backend matches
// and no explicit emulator backend feature is selected, the crate builds with
// no `sf_backend_*` cfg and later code reports the backend as unavailable.
//
// Feature → cfg mapping:
//   (derived)      → sf_has_std          (set whenever any feature that needs libstd is on:
//                                          wasi, call-trace, guard-pages)
//   (derived)      → sf_has_guard_pages  (guard-pages + jit + native 64-bit backend + supported OS)
//   (derived)      → sf_has_debug_regions (set when any consumer of DebugRegion is compiled
//                                          in: sf_ir_dump or sf_jitdump)
//   (derived)      → sf_has_simd         (x64 with SSE2, arm64 with NEON)
//   jit            → sf_jit
//   backend-emu64  → sf_backend_emu64
//   backend-emu32  → sf_backend_emu32
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
    "sf_backend_arm64",
    "sf_backend_armv7a",
    "sf_backend_thumbm",
    "sf_backend_x64",
    "sf_backend_emu64",
    "sf_backend_emu32",
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
    "sf_wasi_host",
    "sf_module_validator",
    "sf_has_simd",
    "sf_call_trace",
    "sf_fp_dp",
    "sf_ir_dump",
    "sf_jitdump",
];

fn main() {
    for name in DECLARED_CFGS {
        println!("cargo::rustc-check-cfg=cfg({name})");
    }

    emit_backend_cfgs();
    emit_simd_cfg();
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedBackend {
    Arm64,
    Armv7a,
    ThumbM,
    X64,
    Emu64,
    Emu32,
}

fn selected_backend() -> Option<SelectedBackend> {
    let emu64 = env::var_os("CARGO_FEATURE_BACKEND_EMU64").is_some();
    let emu32 = env::var_os("CARGO_FEATURE_BACKEND_EMU32").is_some();
    match (emu64, emu32) {
        // `--all-features` turns on both emulator selectors. Treat that as
        // "no explicit emulator override" so the host-native backend still
        // typechecks under the broadest feature sweep.
        (true, true) => {}
        (true, false) => return Some(SelectedBackend::Emu64),
        (false, true) => return Some(SelectedBackend::Emu32),
        (false, false) => {}
    }

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    match arch.as_str() {
        "aarch64" => Some(SelectedBackend::Arm64),
        "arm" => {
            if target.starts_with("thumbv") {
                Some(SelectedBackend::ThumbM)
            } else {
                Some(SelectedBackend::Armv7a)
            }
        }
        "x86_64" => Some(SelectedBackend::X64),
        _ => None,
    }
}

fn emit_backend_cfgs() {
    // Source code uses `sf_backend_*` cfgs instead of raw `target_arch = ...`.
    // Exactly one backend cfg is set when the build selects a supported
    // backend.
    match selected_backend() {
        Some(SelectedBackend::Arm64) => println!("cargo:rustc-cfg=sf_backend_arm64"),
        Some(SelectedBackend::Armv7a) => {
            println!("cargo:rustc-cfg=sf_backend_armv7a");
            // ARMv7-A targets with IDIV always have VFPv3-D16+.
            println!("cargo:rustc-cfg=sf_fp_dp");
            // Opt-in Thumb-2 emit path for encoder validation under
            // the armv7-A qemu harness. A32 Rust ↔ Thumb-2 JIT code
            // bridge via ARM/Thumb interworking at BX/BLX boundaries.
            if env::var_os("CARGO_FEATURE_THUMB2_TEST").is_some() {
                println!("cargo:rustc-cfg=sf_arm32_isa_thumb");
            }
        }
        Some(SelectedBackend::ThumbM) => {
            println!("cargo:rustc-cfg=sf_backend_thumbm");
            println!("cargo:rustc-cfg=sf_arm32_isa_thumb");
            // sf_fp_dp stays off on thumbm: no M-profile FPU offers
            // double-precision arithmetic (FPv5-SP-D16 is SP-only, and
            // no DP variant exists for M-profile). Wasm f64 on thumbm
            // requires a separate legalization story and is out of
            // scope for the initial bring-up.
        }
        Some(SelectedBackend::X64) => println!("cargo:rustc-cfg=sf_backend_x64"),
        Some(SelectedBackend::Emu64) => println!("cargo:rustc-cfg=sf_backend_emu64"),
        Some(SelectedBackend::Emu32) => println!("cargo:rustc-cfg=sf_backend_emu32"),
        None => {}
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

fn emit_simd_cfg() {
    match selected_backend() {
        Some(SelectedBackend::X64) => {
            require_target_feature("sse2", "x86_64 SIMD support requires SSE2");
            println!("cargo:rustc-cfg=sf_has_simd");
        }
        Some(SelectedBackend::Arm64) => {
            require_target_feature("neon", "arm64 SIMD support requires NEON");
            println!("cargo:rustc-cfg=sf_has_simd");
        }
        _ => {}
    }
}

fn require_target_feature(feature: &str, message: &str) {
    if target_features().iter().any(|enabled| *enabled == feature) {
        return;
    }

    let target = env::var("TARGET").unwrap_or_default();
    let feature_list = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    panic!("{message} (target={target}, target_feature={feature_list})");
}

fn target_features() -> Vec<String> {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .filter(|feature| !feature.is_empty())
        .map(str::to_owned)
        .collect()
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
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let backend = selected_backend();
    let os_ok = matches!(os.as_str(), "macos" | "linux")
        || (os == "windows" && matches!(backend, Some(SelectedBackend::X64)));
    if os_ok && matches!(backend, Some(SelectedBackend::X64 | SelectedBackend::Arm64)) {
        println!("cargo:rustc-cfg=sf_has_guard_pages");
    }
}
