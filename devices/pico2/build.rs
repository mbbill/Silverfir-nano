//! Copy the arch-appropriate `memory.x` into OUT_DIR for the active
//! linker script, then build the `wasm-demo/` sub-crate as a
//! `wasm32-unknown-unknown` release artifact so `demo_host.rs`
//! can `include_bytes!` the result.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    println!("cargo:rustc-link-search={}", out.display());

    // Pick the per-arch memory.x:
    //   ARM (target_arch = "arm")     -> memory.arm.x  (cortex-m-rt supplement)
    //   RV32 (target_arch = "riscv32") -> memory.rv.x  (standalone linker script)
    // Both files share the physical FLASH/RAM regions. See the file headers
    // for details.
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let memory_x: &[u8] = match arch.as_str() {
        "arm" => include_bytes!("memory.arm.x"),
        "riscv32" => include_bytes!("memory.rv.x"),
        other => panic!("sf-nano-pico2 has no memory.x for target_arch = \"{other}\""),
    };
    let mut f = File::create(out.join("memory.x")).unwrap();
    f.write_all(memory_x).unwrap();

    println!("cargo:rerun-if-changed=memory.arm.x");
    println!("cargo:rerun-if-changed=memory.rv.x");
    println!("cargo:rerun-if-changed=build.rs");

    build_wasm_demo(&out);
}

fn build_wasm_demo(out_dir: &PathBuf) {
    // Invoke cargo on the sub-crate. Its own `[workspace]` stub keeps
    // it isolated from this firmware's dep graph; --manifest-path also
    // keeps cwd independent of where the firmware build is running
    // from.
    let crate_root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let manifest = crate_root.join("wasm-demo/Cargo.toml");

    // Explicitly set RUSTFLAGS for the wasm build. `.cargo/config.toml`
    // inside wasm-demo/ is NOT auto-discovered when cargo runs with
    // --manifest-path from a different CWD — config lookup follows
    // CWD (and CARGO_HOME), not the manifest path. Setting RUSTFLAGS
    // via the environment is the reliable path.
    //
    // -zstack-size=16384 caps the Wasm call stack at 16 KiB so the
    // module's initial linear memory stays at 1 page instead of the
    // 17-page default (1 MiB stack + data). See the host-side
    // RuntimeConfig's wasm_memory_max_pages quota.
    let wasm_rustflags = "-C link-arg=-zstack-size=16384";
    let demo_feature = selected_demo_feature();
    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.arg("build")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target")
        .arg("wasm32-unknown-unknown")
        .arg("--release")
        .arg("--no-default-features")
        .arg("--features")
        .arg(demo_feature)
        .env("RUSTFLAGS", wasm_rustflags)
        // Clear the parent build's encoded rustflags (contains
        // `-C target-cpu=cortex-m33`, which would be rejected for wasm32).
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        // current_dir lets cargo also pick up wasm-demo/.cargo/config.toml
        // for anyone running this build manually.
        .current_dir(crate_root.join("wasm-demo"));

    let status = cmd.status().expect("failed to launch cargo for wasm-demo");
    if !status.success() {
        panic!("wasm-demo cargo build failed (exit: {status:?})");
    }

    let wasm = crate_root
        .join("wasm-demo")
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("sf_nano_pico2_wasm_demo.wasm");
    std::fs::copy(&wasm, out_dir.join("demo.wasm"))
        .unwrap_or_else(|e| panic!("copy {wasm:?}: {e}"));

    // Track the whole sub-crate directory. A narrower list of files
    // missed updates to `.cargo/config.toml` — we want any change
    // inside wasm-demo/ to trigger a rebuild here.
    println!("cargo:rerun-if-changed=wasm-demo");
    println!("cargo:rerun-if-changed=src/kernels");
}

fn selected_demo_feature() -> &'static str {
    let mandelbrot = std::env::var_os("CARGO_FEATURE_DEMO_MANDELBROT").is_some();
    let cube = std::env::var_os("CARGO_FEATURE_DEMO_CUBE").is_some();
    match (mandelbrot, cube) {
        (true, false) => "demo-mandelbrot",
        (false, true) => "demo-cube",
        (true, true) => panic!("select exactly one demo feature"),
        (false, false) => panic!("select one demo feature"),
    }
}
