//! Build the selected Wasm guest and copy it into OUT_DIR for `include_bytes!`.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MODE_WASM");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_MODE_WASM").is_none() {
        return;
    }

    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    build_wasm_demo(&out);
}

fn build_wasm_demo(out_dir: &PathBuf) {
    let crate_root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let manifest = crate_root.join("wasm-demo/Cargo.toml");

    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.arg("build")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target")
        .arg("wasm32-unknown-unknown")
        .arg("--release")
        .arg("--no-default-features")
        .arg("--features")
        .arg(selected_demo_feature())
        .env("RUSTFLAGS", "-C link-arg=-zstack-size=16384")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
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
        .join("waveshare_esp32_c6_wasm_demo.wasm");
    std::fs::copy(&wasm, out_dir.join("demo.wasm"))
        .unwrap_or_else(|e| panic!("copy {wasm:?}: {e}"));

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
