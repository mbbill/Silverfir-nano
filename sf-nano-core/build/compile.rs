// C compilation utilities
// Handles compilation of fast interpreter trampoline and handler C code

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;

/// Track all C/H files in a directory for cargo rebuild.
fn track_c_files_in_dir(dir: &str) {
    println!("cargo:rerun-if-changed={}", dir);
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension() == Some(OsStr::new("c"))
                || path.extension() == Some(OsStr::new("h"))
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

/// Verify preserve_none register mapping matches JIT's Reg enum.
/// Only runs for micro-jit feature on aarch64 targets.
pub fn verify_preserve_none_abi() {
    if env::var("CARGO_FEATURE_MICRO_JIT").is_err() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        return;
    }

    let c_root = "src/vm/interp/trampoline";
    let src = format!("{}/verify_abi.c", c_root);
    if fs::metadata(&src).is_err() {
        println!(
            "cargo:warning=preserve_none ABI probe not found in new vm tree, skipping verification"
        );
        return;
    }
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let asm_path = PathBuf::from(&out_dir).join("verify_abi.s");

    println!("cargo:rerun-if-changed={}", src);

    let target = env::var("TARGET").unwrap_or_default();

    let status = std::process::Command::new("clang")
        .args([
            "-target",
            &target,
            "-S",
            "-O2",
            "-I",
            c_root,
            "-o",
            asm_path.to_str().unwrap(),
            &src,
        ])
        .status()
        .expect("Failed to run clang for ABI verification");

    if !status.success() {
        panic!("Failed to compile ABI verification probe");
    }

    let asm = fs::read_to_string(&asm_path).expect("Failed to read ABI verification assembly");

    // Expected mapping: arg[i] → register
    let expected = [
        "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x0", "x1",
    ];
    let arg_names = [
        "ctx", "pc", "fp", "l0", "l1", "l2", "t0", "t1", "t2", "t3", "nh",
    ];

    // Extract str instructions from the probe function
    let mut in_probe = false;
    let mut str_regs: Vec<String> = Vec::new();

    for line in asm.lines() {
        let trimmed = line.trim();
        if trimmed.contains("__jit_abi_probe") && trimmed.contains(':') {
            in_probe = true;
            continue;
        }
        if in_probe && (trimmed == "ret" || trimmed.starts_with(".cfi_endproc")) {
            break;
        }
        if in_probe {
            if let Some(reg) = parse_str_reg(trimmed) {
                str_regs.push(reg);
            }
        }
    }

    if str_regs.len() < expected.len() {
        panic!(
            "\n\npreserve_none ABI VERIFICATION FAILED\n\
             Expected {} str instructions in __jit_abi_probe, found {}.\n\
             Assembly output: {}\n",
            expected.len(),
            str_regs.len(),
            asm_path.display()
        );
    }

    for (i, (got, want)) in str_regs.iter().zip(expected.iter()).enumerate() {
        if got != want {
            panic!(
                "\n\npreserve_none ABI CHANGED — BUILD ABORTED\n\n\
                 Argument '{}' (index {}): expected register {}, got {}.\n\n\
                 The preserve_none calling convention register assignment has changed.\n\
                 Update jit/reg.rs Reg enum values to match the new mapping.\n\
                 See docs/MICRO_JIT.md §ARM64 Register Map.\n\
                 Assembly output: {}\n",
                arg_names[i],
                i,
                want,
                got,
                asm_path.display()
            );
        }
    }

}

/// Parse a `str xN, [...]` instruction and return the source register name.
fn parse_str_reg(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with("str\t") && !trimmed.starts_with("str ") {
        return None;
    }
    let after_str = trimmed[3..].trim();
    let reg_end = after_str.find(',')?;
    let reg = after_str[..reg_end].trim();
    if reg.starts_with('x') || reg.starts_with('w') {
        // Normalize wN to xN for comparison
        let normalized = if reg.starts_with('w') {
            format!("x{}", &reg[1..])
        } else {
            reg.to_string()
        };
        Some(normalized)
    } else {
        None
    }
}

/// Compile the fast interpreter trampoline
pub fn compile_fast_trampoline(out_dir: &str) {
    let c_root = "src/vm/interp/trampoline";
    let c_handlers_root = "src/vm/interp/handlers_c";
    let fast_trampoline_src = format!("{}/vm_trampoline.c", c_root);

    if fs::metadata(&fast_trampoline_src).is_err() {
        println!("cargo:warning=Fast interpreter C trampoline not found, skipping compilation");
        return;
    }

    let mut build = cc::Build::new();
    build.compiler("clang");

    let fast_interp_out = PathBuf::from(out_dir).join("fast_interp");

    build
        .file(format!("{}/vm_trampoline.c", c_root))
        .include(c_root)
        .include(out_dir)
        .include(&fast_interp_out);

    let is_debug = env::var("PROFILE").unwrap_or_default() == "debug";
    let has_debug_info = env::var("CARGO_CFG_DEBUG_ASSERTIONS").is_ok()
        || env::var("DEBUG").as_deref() == Ok("true")
        || env::var("DEBUG").as_deref() == Ok("2");

    if !is_debug {
        build.define("NDEBUG", None);
    }

    if env::var("CARGO_FEATURE_FUSION").is_ok() {
        build.define("FUSION_ENABLED", None);
    }

    if env::var("CARGO_FEATURE_TRACE").is_ok() {
        build.define("FAST_TRACE_ENABLED", None);
    }

    if env::var("CARGO_FEATURE_FUNCTION_TRACE").is_ok() {
        build.define("FUNCTION_TRACE_ENABLED", None);
    }

    if env::var("CARGO_FEATURE_PROFILE").is_ok() {
        build.define("FAST_PROFILE_ENABLED", None);
    }

    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build
            .compiler("clang-cl")
            .archiver("llvm-lib")
            .flag_if_supported("/O2")
            .flag_if_supported("/wd4100")
            .flag("/clang:-flto=thin")
            .flag("/clang:-fomit-frame-pointer")
            .flag("/clang:-foptimize-sibling-calls")
            .flag("/clang:-Wno-unused-parameter");

        if has_debug_info {
            build.flag("/Zi");
        }

        build.compile("libvm_trampoline.a");
    } else {
        build
            .flag("-O3")
            .flag("-ffast-math")
            .flag("-fno-finite-math-only")
            .flag("-foptimize-sibling-calls")
            .flag("-march=native")
            .flag("-Wno-unused-parameter");

        if has_debug_info {
            build.flag("-g").flag("-fno-omit-frame-pointer");
        } else {
            build.flag("-fomit-frame-pointer");
        }

        build.compile("vm_trampoline");
    }

    track_c_files_in_dir(c_root);
    track_c_files_in_dir(c_handlers_root);
}
