use super::{CompileConfig, HOT_LOCAL_COUNT, hot_local};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HotLocalPlan {
    raw: [Option<u32>; HOT_LOCAL_COUNT],
    effective: [Option<u32>; HOT_LOCAL_COUNT],
}

impl HotLocalPlan {
    pub fn analyze(code: &[u8], frame_size: usize, config: CompileConfig) -> Self {
        let raw = hot_local::find_hot_locals(code, frame_size, config);
        let effective = hot_local::compute_effective_indices(&raw, frame_size);
        Self { raw, effective }
    }

    #[inline]
    pub fn raw(self) -> [Option<u32>; HOT_LOCAL_COUNT] {
        self.raw
    }

    #[inline]
    pub fn effective(self) -> [Option<u32>; HOT_LOCAL_COUNT] {
        self.effective
    }

    #[inline]
    pub fn hot_mask(self) -> [bool; HOT_LOCAL_COUNT] {
        [
            self.effective[0].is_some(),
            self.effective[1].is_some(),
            self.effective[2].is_some(),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompilePlan {
    config: CompileConfig,
    hot_locals: HotLocalPlan,
}

impl CompilePlan {
    pub fn for_config(code: &[u8], frame_size: usize, config: CompileConfig) -> Self {
        Self {
            config,
            hot_locals: HotLocalPlan::analyze(code, frame_size, config),
        }
    }

    #[inline]
    pub fn config(self) -> CompileConfig {
        self.config
    }

    #[inline]
    pub fn hot_locals(self) -> HotLocalPlan {
        self.hot_locals
    }
}
