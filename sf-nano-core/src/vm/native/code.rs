//! Native compiled code object.
//!
//! This should own:
//! - executable code
//! - helper metadata
//! - patch metadata
//! - finalized native execution metadata during bring-up
//!
//! It must not own an interpreter-shaped descriptor stream.

use alloc::{boxed::Box, vec::Vec};

use super::{
    code_buf::CodeBuffer,
    entry::NativeEntry,
    helper_meta::{HelperMetadataArena, HelperMetadataRecord},
    ir::NativeProgram,
};

/// Direct-call patch site that must be finalized to a native entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectCallPatch {
    pub callee_func_idx: u32,
    pub entry_branch_inst_offset: u32,
    pub entry_branch_reg_offset: u32,
    pub entry_literal_offset: u32,
    pub frame_slots_literal_offset: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeCodeRegion {
    pub start: u32,
    pub len: u32,
}

/// Native compiled code object.
///
/// This is intentionally *not* a descriptor stream. It owns executable bytes
/// plus the side metadata required by native wrappers and patching. During
/// native bring-up it also carries finalized native execution metadata so the
/// native runtime can execute the native IR without falling back to the
/// interpreter instruction stream.
#[derive(Debug, Default)]
pub struct NativeCode {
    pub entry: Option<NativeEntry>,
    pub internal_entry: Option<NativeEntry>,
    pub text: Box<[u8]>,
    pub region: Option<NativeCodeRegion>,
    pub executable: Option<CodeBuffer>,
    pub helper_metadata: Box<[HelperMetadataRecord]>,
    pub direct_call_patches: Box<[DirectCallPatch]>,
    pub program: Option<NativeProgram>,
}

impl NativeCode {
    #[inline]
    pub fn helper_metadata(&self) -> &[HelperMetadataRecord] {
        &self.helper_metadata
    }

    #[inline]
    pub fn direct_call_patches(&self) -> &[DirectCallPatch] {
        &self.direct_call_patches
    }

    #[inline]
    pub fn program(&self) -> Option<&NativeProgram> {
        self.program.as_ref()
    }

    #[inline]
    pub fn internal_entry(&self) -> Option<NativeEntry> {
        self.internal_entry
    }

    #[inline]
    pub fn region(&self) -> Option<NativeCodeRegion> {
        self.region
    }

    #[inline]
    pub fn from_parts(
        entry: Option<NativeEntry>,
        internal_entry: Option<NativeEntry>,
        text: Vec<u8>,
        region: Option<NativeCodeRegion>,
        executable: Option<CodeBuffer>,
        helper_metadata: HelperMetadataArena,
        direct_call_patches: Vec<DirectCallPatch>,
        program: Option<NativeProgram>,
    ) -> Self {
        Self {
            entry,
            internal_entry,
            text: text.into_boxed_slice(),
            region,
            executable,
            helper_metadata: helper_metadata.into_boxed_slice(),
            direct_call_patches: direct_call_patches.into_boxed_slice(),
            program,
        }
    }

    #[inline]
    pub fn build_cache(
        &self,
        params_len: usize,
        locals_len: usize,
        results_len: usize,
    ) -> NativeCodeCache {
        NativeCodeCache {
            entry: self.entry,
            internal_entry: self.internal_entry,
            params_len,
            locals_len,
            results_len,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NativeCodeCache {
    entry: Option<NativeEntry>,
    internal_entry: Option<NativeEntry>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
}

impl Default for NativeCodeCache {
    fn default() -> Self {
        Self {
            entry: None,
            internal_entry: None,
            params_len: 0,
            locals_len: 0,
            results_len: 0,
        }
    }
}

unsafe impl Send for NativeCodeCache {}
unsafe impl Sync for NativeCodeCache {}

impl NativeCodeCache {
    #[inline(always)]
    pub fn is_compiled(&self) -> bool {
        self.entry.is_some()
    }

    #[inline(always)]
    pub fn entry(&self) -> NativeEntry {
        self.entry.expect("native code cache missing entry")
    }

    #[inline(always)]
    pub fn internal_entry(&self) -> Option<NativeEntry> {
        self.internal_entry
    }

    #[inline(always)]
    pub fn params_len(&self) -> usize {
        self.params_len
    }

    #[inline(always)]
    pub fn locals_len(&self) -> usize {
        self.locals_len
    }

    #[inline(always)]
    pub fn results_len(&self) -> usize {
        self.results_len
    }
}

pub fn create_native_code(
    entry: Option<NativeEntry>,
    internal_entry: Option<NativeEntry>,
    text: Vec<u8>,
    region: Option<NativeCodeRegion>,
    executable: Option<CodeBuffer>,
    helper_metadata: HelperMetadataArena,
    direct_call_patches: Vec<DirectCallPatch>,
    program: Option<NativeProgram>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
) -> (NativeCode, NativeCodeCache) {
    let code = NativeCode::from_parts(
        entry,
        internal_entry,
        text,
        region,
        executable,
        helper_metadata,
        direct_call_patches,
        program,
    );
    let cache = code.build_cache(params_len, locals_len, results_len);
    (code, cache)
}
