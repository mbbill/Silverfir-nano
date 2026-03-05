// Fusion code generator: produces fused C handlers and IR-level pattern matching
// from the [[fused]] entries in handlers_fused.toml.
//
// Outputs:
//   gen_fusion_c        — fast_fused_handlers.inc (C handler bodies)
//   gen_fusion_ir_match — fast_fusion_ir_match.rs (IR-level pattern matching)

use super::types::HandlersFile;
use std::fs;
use std::path::PathBuf;

pub fn generate(handlers: &HandlersFile, out_dir: &PathBuf) {
    let fused = &handlers.fused;
    let categories = handlers.category_map();

    // Output 1: fast_fused_handlers.inc (C handler implementations)
    let fused_c = if fused.is_empty() {
        "// No fused handlers (handlers_fused.toml not found)\n".to_string()
    } else {
        super::gen_fusion_c::generate(fused, &categories)
    };
    let fused_c_path = out_dir.join("fast_fused_handlers.inc");
    fs::write(&fused_c_path, fused_c)
        .unwrap_or_else(|_| panic!("Failed to write {:?}", fused_c_path));

    // Output 2: fast_fusion_ir_match.rs (IR-level fusion pattern matching)
    let fusion_ir = if fused.is_empty() {
        super::gen_fusion_ir_match::generate_empty()
    } else {
        super::gen_fusion_ir_match::generate(fused, &categories)
    };
    let fusion_ir_path = out_dir.join("fast_fusion_ir_match.rs");
    fs::write(&fusion_ir_path, fusion_ir)
        .unwrap_or_else(|_| panic!("Failed to write {:?}", fusion_ir_path));
}
