use std::{env, fs, path::Path, process::Command};

const WASI_TESTSUITE_URL: &str =
    "https://github.com/WebAssembly/wasi-testsuite/archive/96bcf61cf88023ca547d60f8b04c49f3d43d2838.tar.gz";
const TESTSUITE_VERSION_FILE: &str = "wasi_testsuite_version.txt";
const TESTSUITE_VERSION: &str = "96bcf61cf88023ca547d60f8b04c49f3d43d2838";

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let cargo_target_dir = Path::new(&out_dir)
        .ancestors()
        .find(|path| path.file_name().is_some_and(|name| name == "target"))
        .unwrap_or_else(|| Path::new("target"));

    let testsuite_dir = cargo_target_dir.join("wasi-testsuite");
    let version_file = testsuite_dir.join(TESTSUITE_VERSION_FILE);

    // Check if testsuite already exists and is up to date
    let need_download = if testsuite_dir.exists() && version_file.exists() {
        match fs::read_to_string(&version_file) {
            Ok(existing_version) => {
                let version_matches = existing_version.trim() == TESTSUITE_VERSION;
                if version_matches {
                    println!(
                        "cargo:rustc-env=WASI_TESTSUITE_DIR={}",
                        testsuite_dir.display()
                    );
                    return;
                }
                true
            }
            Err(_) => true,
        }
    } else {
        true
    };

    if need_download {
        println!("cargo:warning=Downloading WASI testsuite for sf-nano...");
        download_and_extract_testsuite(&testsuite_dir).expect("Failed to download WASI testsuite");
        fs::write(&version_file, TESTSUITE_VERSION).expect("Failed to write version file");
        println!("cargo:warning=WASI testsuite downloaded and extracted");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    if version_file.exists() {
        println!("cargo:rerun-if-changed={}", version_file.display());
    }

    println!(
        "cargo:rustc-env=WASI_TESTSUITE_DIR={}",
        testsuite_dir.display()
    );
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

    let tar_path = temp_dir.join("wasi-testsuite.tar.gz");

    download_file(WASI_TESTSUITE_URL, &tar_path)?;
    extract_tar_gz(&tar_path, &temp_dir)?;

    // Find the extracted directory
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

    Err("Neither curl nor wget found. Please install one to download the WASI testsuite.".into())
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

    Err("tar command not found. Please install tar to extract the WASI testsuite.".into())
}
