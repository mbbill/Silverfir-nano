// sf-nano-core build script
// Emits platform-derived cfgs and verifies LLVM toolchain compatibility.

use std::env;
use std::process::Command;

fn main() {
    check_llvm_version_compatibility();
    emit_guard_pages_cfg();
}

fn emit_guard_pages_cfg() {
    println!("cargo::rustc-check-cfg=cfg(has_guard_pages)");
    let dominated = env::var_os("CARGO_FEATURE_GUARD_PAGES").is_some()
        && env::var_os("CARGO_FEATURE_MICRO_JIT").is_some();
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
        println!("cargo:rustc-cfg=has_guard_pages");
    }
}

fn check_llvm_version_compatibility() {
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }

    let rustc_llvm = get_rustc_llvm_version();
    let clang_llvm = get_clang_llvm_version();

    match (rustc_llvm, clang_llvm) {
        (Some(rustc_ver), Some(clang_ver)) if rustc_ver != clang_ver => {
            panic!(
                "\n\nLLVM VERSION MISMATCH\n\n\
                rustc uses LLVM {rustc_ver}, but clang uses LLVM {clang_ver}.\n\n\
                Cross-language LTO requires matching LLVM major versions.\n"
            );
        }
        _ => {}
    }
}

fn get_rustc_llvm_version() -> Option<u32> {
    let output = Command::new("rustc").args(["-vV"]).output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(version_str) = line.strip_prefix("LLVM version:") {
            return version_str.trim().split('.').next()?.parse().ok();
        }
    }
    None
}

fn get_clang_llvm_version() -> Option<u32> {
    let output = Command::new("clang-cl")
        .args(["--version"])
        .output()
        .or_else(|_| Command::new("clang").args(["--version"]).output())
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("clang version ") {
            return rest.split('.').next()?.parse().ok();
        }
    }
    None
}
