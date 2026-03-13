// sf-nano-core build script
// Generates Fast interpreter code, then compiles C trampolines

#[path = "build/compile.rs"]
mod compile;

#[path = "build/fast_interp/mod.rs"]
mod fast_interp;

use std::process::Command;
use std::{env, path::PathBuf};

fn main() {
    check_llvm_version_compatibility();

    if env::var_os("CARGO_FEATURE_INTERP").is_some() {
        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let out_path = PathBuf::from(&out_dir);

        // Generate fast interpreter code from handlers.toml.
        fast_interp::generate(&out_path);

        // Verify preserve_none ABI for the interpreter trampoline.
        compile::verify_preserve_none_abi();

        // Compile fast interpreter C trampoline.
        compile::compile_fast_trampoline(&out_dir);
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
