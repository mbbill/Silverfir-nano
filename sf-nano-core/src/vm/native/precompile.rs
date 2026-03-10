//! Native precompile entry.

use crate::error::WasmError;

use super::{
    build::NativeBuildBundle,
    finalizer::{self, NativeFinalized},
    resolve,
};

/// Native precompile result.
#[derive(Debug, Default)]
pub struct NativePrecompiled {
    pub finalized: Option<NativeFinalized>,
}

pub fn precompile_native(bundle: NativeBuildBundle) -> Result<NativePrecompiled, WasmError> {
    let resolved = resolve::resolve_native(&bundle.lir);
    let finalized = finalizer::finalize_native(&bundle.lir, &resolved);

    Ok(NativePrecompiled {
        finalized: Some(finalized),
    })
}
