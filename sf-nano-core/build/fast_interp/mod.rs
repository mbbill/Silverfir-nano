// Fast interpreter build module
// Generates code from handlers.toml (merged handler + encoding definitions)

pub mod gen_tos_config_h;
pub mod tos_config;
pub mod types;
#[macro_use]
pub mod code_writer;
pub mod gen_c_wrappers;
pub mod gen_encoding;
pub mod gen_extern_decl;
pub mod gen_fusion;
pub mod gen_fusion_c;
pub mod gen_fusion_ir_match;
pub mod gen_handler_lookup;
pub mod gen_handler_map;
pub mod gen_handler_names;
pub mod gen_ir_resolve;
pub mod op_classify;

use std::fs;
use std::path::PathBuf;

use types::{FusedFile, HandlersFile};

/// Generate all fast interpreter code from handlers.toml + handlers_fused.toml
pub fn generate(out_dir: &PathBuf) {
    let fast_dir = "src/vm/interp";
    let handlers_toml_path = format!("{}/handlers.toml", fast_dir);
    let fused_toml_path = format!("{}/handlers_fused.toml", fast_dir);

    // Create output directory
    let fast_out = out_dir.join("fast_interp");
    fs::create_dir_all(&fast_out).expect("Failed to create fast_interp output directory");

    // Generate TOS config header first (no dependencies)
    gen_tos_config_h::generate(&fast_out);

    // Check if handler definition file exists
    if fs::metadata(&handlers_toml_path).is_err() {
        println!("cargo:warning=Fast interpreter handlers.toml not found, skipping generation");
        return;
    }

    // Parse handler definitions
    let mut handlers = parse_handlers(&handlers_toml_path);

    // Merge auto-discovered fused instructions (if file exists)
    if let Ok(content) = fs::read_to_string(&fused_toml_path) {
        let fused: FusedFile =
            toml::from_str(&content).expect("Failed to parse handlers_fused.toml");
        handlers.fused = fused.fused;
    }

    // Validate after parsing + merging
    handlers.validate();

    // Generate all outputs from single source
    gen_encoding::generate(&handlers, &fast_out);
    gen_c_wrappers::generate(&handlers, &fast_out);
    gen_extern_decl::generate(&handlers, &fast_out);
    gen_handler_map::generate(&handlers, &fast_out);
    gen_handler_names::generate(&handlers, &fast_out);
    gen_handler_lookup::generate(&handlers, &fast_out);
    gen_fusion::generate(&handlers, &fast_out);
    gen_ir_resolve::generate(&handlers, &fast_out);

    // Track dependencies
    println!("cargo:rerun-if-changed={}", handlers_toml_path);
    println!("cargo:rerun-if-changed={}", fused_toml_path);
}

fn parse_handlers(path: &str) -> HandlersFile {
    let content = fs::read_to_string(path).expect("Failed to read handlers.toml");
    toml::from_str(&content).expect("Failed to parse handlers.toml")
}
