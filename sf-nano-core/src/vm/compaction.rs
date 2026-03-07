#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionDisposition {
    Keep,
    RedirectBranchTarget,
    InternalOnly,
}

impl CompactionDisposition {
    #[inline]
    pub fn is_kept(self) -> bool {
        matches!(self, Self::Keep)
    }

    #[inline]
    pub fn may_redirect_branch_target(self) -> bool {
        matches!(self, Self::RedirectBranchTarget)
    }
}
