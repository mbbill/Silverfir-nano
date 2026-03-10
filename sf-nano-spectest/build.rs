use std::{env, fs, path::Path, process::Command};

// Pin to the last WASM 2.0 commit before wasm-3.0 tests were merged (Oct 30, 2024)
const TESTSUITE_URL: &str =
    "https://github.com/WebAssembly/testsuite/archive/eeb6dac81be81151da0958ed955aba91230c1f25.tar.gz";
const TESTSUITE_VERSION_FILE: &str = "testsuite_version.txt";
const TESTSUITE_VERSION: &str = "wasm2.0-eeb6dac";

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let cargo_target_dir = Path::new(&out_dir)
        .ancestors()
        .find(|path| path.file_name().is_some_and(|name| name == "target"))
        .unwrap_or_else(|| Path::new("target"));

    let testsuite_dir = cargo_target_dir.join("webassembly-testsuite-2.0");
    let version_file = testsuite_dir.join(TESTSUITE_VERSION_FILE);

    let need_download = if testsuite_dir.exists() && version_file.exists() {
        match fs::read_to_string(&version_file) {
            Ok(existing_version) => existing_version.trim() != TESTSUITE_VERSION,
            Err(_) => true,
        }
    } else {
        true
    };

    if need_download {
        println!("cargo:warning=Downloading WebAssembly testsuite...");
        download_and_extract_testsuite(&testsuite_dir).expect("Failed to download testsuite");
        fs::write(&version_file, TESTSUITE_VERSION).expect("Failed to write version file");
        println!("cargo:warning=WebAssembly testsuite downloaded and extracted");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    if version_file.exists() {
        println!("cargo:rerun-if-changed={}", version_file.display());
    }

    println!("cargo:rustc-env=TESTSUITE_DIR={}", testsuite_dir.display());
}

fn download_and_extract_testsuite(testsuite_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = testsuite_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    if testsuite_dir.exists() {
        fs::remove_dir_all(testsuite_dir)?;
    }

    let temp_dir = testsuite_dir.with_extension("tmp");
    fs::create_dir_all(&temp_dir)?;

    let tar_path = temp_dir.join("testsuite.tar.gz");

    download_file(TESTSUITE_URL, &tar_path)?;
    extract_tar_gz(&tar_path, &temp_dir)?;

    let entries: Vec<_> = fs::read_dir(&temp_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().ok().is_some_and(|ft| ft.is_dir()))
        .collect();

    if let Some(extracted_dir) = entries.first() {
        fs::rename(extracted_dir.path(), testsuite_dir)?;
    } else {
        return Err("No directory found in extracted archive".into());
    }

    fs::remove_dir_all(temp_dir)?;

    Ok(())
}

fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if Command::new("curl")
        .args(["-L", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .is_ok_and(|status| status.success())
    {
        return Ok(());
    }

    if Command::new("wget")
        .args(["-O"])
        .arg(dest)
        .arg(url)
        .status()
        .is_ok_and(|status| status.success())
    {
        return Ok(());
    }

    Err(
        "Neither curl nor wget found. Please install one to download the WebAssembly testsuite."
            .into(),
    )
}

fn extract_tar_gz(archive_path: &Path, dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if Command::new("tar")
        .args(["-xzf"])
        .arg(archive_path)
        .args(["-C"])
        .arg(dest_dir)
        .status()
        .is_ok_and(|status| status.success())
    {
        return Ok(());
    }

    Err("tar command not found. Please install tar to extract the WebAssembly testsuite.".into())
}
