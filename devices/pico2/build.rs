//! Copy `memory.x` into OUT_DIR so cortex-m-rt's `link.x` can find it,
//! then build the `wasm-kernel/` sub-crate as a `wasm32-unknown-unknown`
//! release artifact so `mandelbrot_wasm.rs` can `include_bytes!` the
//! result.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    println!("cargo:rustc-link-search={}", out.display());

    let memory_x = include_bytes!("memory.x");
    let mut f = File::create(out.join("memory.x")).unwrap();
    f.write_all(memory_x).unwrap();

    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");

    build_wasm_kernel(&out);
}

fn build_wasm_kernel(out_dir: &PathBuf) {
    // Invoke cargo on the sub-crate. Its own `[workspace]` stub keeps
    // it isolated from this firmware's dep graph; --manifest-path also
    // keeps cwd independent of where the firmware build is running
    // from.
    let crate_root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let manifest = crate_root.join("wasm-kernel/Cargo.toml");

    // Explicitly set RUSTFLAGS for the wasm build. `.cargo/config.toml`
    // inside wasm-kernel/ is NOT auto-discovered when cargo runs with
    // --manifest-path from a different CWD — config lookup follows
    // CWD (and CARGO_HOME), not the manifest path. Setting RUSTFLAGS
    // via the environment is the reliable path.
    //
    // -zstack-size=16384 caps the Wasm call stack at 16 KiB so the
    // module's initial linear memory stays at 1 page instead of the
    // 17-page default (1 MiB stack + data). See the host-side
    // RuntimeConfig's wasm_memory_max_pages quota.
    let wasm_rustflags = "-C link-arg=-zstack-size=16384";
    let status = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .arg("build")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target")
        .arg("wasm32-unknown-unknown")
        .arg("--release")
        .env("RUSTFLAGS", wasm_rustflags)
        // Clear the parent build's encoded rustflags (contains
        // `-C target-cpu=cortex-m33`, which would be rejected for wasm32).
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        // current_dir lets cargo also pick up wasm-kernel/.cargo/config.toml
        // for anyone running this build manually.
        .current_dir(crate_root.join("wasm-kernel"))
        .status()
        .expect("failed to launch cargo for wasm-kernel");
    if !status.success() {
        panic!("wasm-kernel cargo build failed (exit: {status:?})");
    }

    let wasm = crate_root
        .join("wasm-kernel")
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("sf_nano_pico2_wasm_kernel.wasm");
    std::fs::copy(&wasm, out_dir.join("kernel.wasm"))
        .unwrap_or_else(|e| panic!("copy {wasm:?}: {e}"));

    // Track the whole sub-crate directory. A narrower list of files
    // missed updates to `.cargo/config.toml` — we want any change
    // inside wasm-kernel/ to trigger a rebuild here.
    println!("cargo:rerun-if-changed=wasm-kernel");
    println!("cargo:rerun-if-changed=src/mandelbrot_kernel.rs");
}
