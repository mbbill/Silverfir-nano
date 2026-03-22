//! Shared debug dump layout helpers.

use alloc::{format, string::String};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DumpRoot {
    pub root: String,
    pub module: String,
}

impl DumpRoot {
    #[inline]
    pub(crate) fn function_dir(&self, func_idx: u32) -> String {
        format!("{}/{}/func_{:04}", self.root, self.module, func_idx)
    }

    #[inline]
    pub(crate) fn lowered_ir_path(&self, func_idx: u32) -> String {
        format!("{}/lowered_ir.txt", self.function_dir(func_idx))
    }

    #[inline]
    pub(crate) fn native_dir(&self, func_idx: u32) -> String {
        format!("{}/native", self.function_dir(func_idx))
    }

    #[inline]
    pub(crate) fn fast_path(&self, func_idx: u32) -> String {
        format!("{}/fast_resolved.txt", self.function_dir(func_idx))
    }
}
