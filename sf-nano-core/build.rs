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
//   riscv32gc*             → sf_backend_riscv32  (RV32GC scalar codegen, sf_fp_dp on)
//   riscv32imac* (etc.)    → sf_backend_riscv32  (no F/D — Hazard3 / Pico 2 RV mode)
//   riscv64gc*             → sf_backend_riscv64  (RV64GC scalar codegen)
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
//   riscv32 × { linux, none }
//   riscv64 × { linux, none }
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
//   (derived)      → sf_has_simd         (arm64 with NEON, x64 with SSE4.1+SSSE3)
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

// The interpreter's dispatch handlers are generated here, at build time,
// and folded into the binary's `.text` by `global_asm!`. These three
// modules are compiled twice — once into the crate, once into this build
// script — so the generator enumerates exactly the handler variant space
// the linker looks up, from one source. Each consumer uses a different
// subset of `layout`, hence the allow.
#[path = "src/vm/interpreter/instr.rs"]
#[allow(dead_code)]
mod instr;
#[allow(dead_code)]
#[path = "interp_gen/mod.rs"]
mod interp_gen;
#[allow(dead_code)]
#[path = "src/vm/interpreter/layout.rs"]
mod layout;

use interp_gen::asm::ObjFmt;
use interp_gen::isa::Isa;

const DECLARED_CFGS: &[&str] = &[
    "sf_backend_arm64",
    "sf_backend_armv7a",
    "sf_backend_thumbm",
    "sf_backend_riscv32",
    "sf_backend_riscv64",
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
    "sf_interp",
    "sf_interp_engine",
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
    emit_interp_engine();
}

/// Generate the interpreter's dispatch engine for this target and write it
/// to `OUT_DIR` as assembler source plus a small facts file. Sets
/// `sf_interp_engine` when a backend exists; without it the interpreter
/// reports cleanly that this target has no engine.
///
/// Deliberately independent of `sf_jit`: build-time generation is what
/// removes the executable-memory requirement, so `--features interp`
/// alone is now a complete engine.
fn emit_interp_engine() {
    println!("cargo:rerun-if-changed=interp_gen");
    println!("cargo:rerun-if-changed=src/vm/interpreter/instr.rs");
    println!("cargo:rerun-if-changed=src/vm/interpreter/layout.rs");
    if env::var_os("CARGO_FEATURE_INTERP").is_none() {
        return;
    }
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let fmt = match os.as_str() {
        "macos" | "ios" => ObjFmt::MachO,
        "windows" => ObjFmt::Coff,
        _ => ObjFmt::Elf,
    };
    let thumb = matches!(selected_backend(), Some(SelectedBackend::ThumbM))
        || (matches!(selected_backend(), Some(SelectedBackend::Armv7a))
            && env::var_os("CARGO_FEATURE_THUMB2_TEST").is_some());
    let mut isa: Box<dyn Isa> = match selected_backend() {
        Some(SelectedBackend::Arm64) => Box::new(interp_gen::arm64::Arm64),
        Some(SelectedBackend::X64) => Box::new(interp_gen::x86_64::X86_64 {
            windows: os == "windows",
        }),
        Some(SelectedBackend::Riscv64) => Box::new(interp_gen::riscv::RiscV {
            xlen: 64,
            // RV64GC's triple contract guarantees F and D.
            fp: true,
        }),
        Some(SelectedBackend::Riscv32) => Box::new(interp_gen::riscv::RiscV {
            xlen: 32,
            // The RV32 handler set is the integer one; f32/f64 route to the
            // shared executor even on `gc` triples.
            fp: false,
        }),
        Some(SelectedBackend::Armv7a) | Some(SelectedBackend::ThumbM) => {
            Box::new(interp_gen::arm32::Arm32 {
                thumb,
                // Both profiles have sdiv/udiv — this project's ARMv7-A
                // targets are the IDIV ones (build.rs already assumes
                // VFPv3-D16, which implies it), and every M-profile core
                // has them. Only the A-profile assembler needs telling.
                idiv_directive: matches!(selected_backend(), Some(SelectedBackend::Armv7a)),
            })
        }
        // Targets with no interpreter backend yet: the crate still builds,
        // and instantiating an interpreter instance fails with a clean
        // error naming the target.
        _ => return,
    };
    let counting = env::var_os("CARGO_FEATURE_INTERP_COUNT").is_some();
    let text = interp_gen::generate(&mut *isa, fmt, thumb, counting);
    let caps = isa.caps();

    let out = env::var("OUT_DIR").expect("OUT_DIR");
    let dir = std::path::Path::new(&out);
    std::fs::write(dir.join("interp_engine.s"), text).expect("write interp_engine.s");
    let cfg = format!(
        "// Generated by build.rs. Facts about the engine in this binary.\n\
         const INTERP_HAS_L1: bool = {};\n\
         const INTERP_HAS_FLOAT_REGS: bool = {};\n\
         const INTERP_FLOAT_PIN_F32: bool = {};\n\
         const INTERP_NATIVE_CALLS: bool = {};\n\
         #[allow(dead_code)]\n\
         const INTERP_PTR_BYTES: u32 = {};\n\
         const INTERP_HANDLER_SLOTS: usize = {};\n",
        caps.has_l1,
        caps.has_float_regs,
        caps.float_pin_f32,
        caps.native_calls,
        caps.ptr_bytes,
        layout::total_slots(),
    );
    std::fs::write(dir.join("interp_engine_cfg.rs"), cfg).expect("write interp_engine_cfg.rs");
    println!("cargo:rustc-cfg=sf_interp_engine");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedBackend {
    Arm64,
    Armv7a,
    ThumbM,
    Riscv32,
    Riscv64,
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
        "riscv32" if target_implies_rv32_supported(&target) => Some(SelectedBackend::Riscv32),
        "riscv64" if target_implies_rv64gc(&target) => Some(SelectedBackend::Riscv64),
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
        Some(SelectedBackend::Riscv32) => {
            println!("cargo:rustc-cfg=sf_backend_riscv32");
            // sf_fp_dp is on for `riscv32gc-*` (F/D guaranteed by triple
            // contract; rustc may not report F/D via target-feature here).
            // For non-`gc` triples (e.g. `riscv32imac-*` on Hazard3 /
            // RP2350 RV mode) the backend stays selected but emits no FP
            // ops, mirroring the integer-only sf_backend_thumbm posture.
            let target = env::var("TARGET").unwrap_or_default();
            if target_implies_rv32gc(&target) {
                println!("cargo:rustc-cfg=sf_fp_dp");
            }
            require_target_feature("m", "rv32 backend requires the RISC-V M extension");
        }
        Some(SelectedBackend::Riscv64) => {
            println!("cargo:rustc-cfg=sf_backend_riscv64");
            // Rust currently reports only a/c/m through CARGO_CFG_TARGET_FEATURE
            // for riscv64gc targets, even though the target codegen/ABI is
            // hard-float and emits ELF attributes for F/D. Use the target
            // triple as the RV64GC contract instead of cfg feature probing.
            println!("cargo:rustc-cfg=sf_fp_dp");
            require_target_feature("m", "rv64 backend requires the RISC-V M extension");
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
    if env::var_os("CARGO_FEATURE_INTERP").is_some() {
        println!("cargo:rustc-cfg=sf_interp");
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
        Some(SelectedBackend::Arm64) => {
            require_target_feature("neon", "arm64 SIMD support requires NEON");
            println!("cargo:rustc-cfg=sf_has_simd");
        }
        Some(SelectedBackend::X64) => {
            // The x64 SIMD lowering emits SSSE3 (`pshufb`) and SSE4.1
            // (`ptest`, `pmovsx*`, `pmovzx*`, `pmulld`, `roundps`/`roundpd`).
            // Those are requirements of the *emitted* code, not of this
            // crate's own compilation, so they are probed against the running
            // CPU in `vm::arch::x86_64::cpu` instead of being asserted here
            // against the build target's baseline. Asserting at build time
            // would force `-C target-feature=+ssse3,+sse4.1` on every x64
            // build via a `.cargo/config.toml` that does not travel with the
            // published crate.
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

fn target_implies_rv64gc(target: &str) -> bool {
    target.starts_with("riscv64gc-")
}

fn target_implies_rv32gc(target: &str) -> bool {
    target.starts_with("riscv32gc-")
}

/// Set of `riscv32-*` target triples the rv32 backend can lower for. The
/// `gc` family is hard-float (sf_fp_dp on); the no-FPU families
/// (`riscv32imac`, `riscv32imc`, `riscv32imafc`-style with no D) build with
/// sf_fp_dp off and emit integer-only Wasm modules.
fn target_implies_rv32_supported(target: &str) -> bool {
    target_implies_rv32gc(target)
        || target.starts_with("riscv32imac-")
        || target.starts_with("riscv32imc-")
        || target.starts_with("riscv32im-")
        || target.starts_with("riscv32i-")
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
