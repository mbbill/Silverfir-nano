/// Current upper bound on backend-managed hot locals encoded by the lowered IR.
///
/// The compile pipeline can activate fewer slots via `CompileConfig::hot_local_count`,
/// but any wider cache model needs a wider lowered representation first.
pub const MAX_HOT_LOCAL_COUNT: usize = 3;
pub const HOT_LOCAL_COUNT: usize = MAX_HOT_LOCAL_COUNT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameLayoutConfig {
    pub metadata_slots: usize,
}

impl FrameLayoutConfig {
    #[inline]
    pub const fn operand_stack_base(self, frame_size: usize) -> usize {
        frame_size + self.metadata_slots
    }

    #[inline]
    pub const fn operand_base_offset(self, frame_size: usize) -> usize {
        self.operand_stack_base(frame_size) * 8
    }

    #[inline]
    pub const fn return_pc_slot(self, frame_size: usize) -> usize {
        frame_size
    }

    #[inline]
    pub const fn saved_fp_slot(self, frame_size: usize) -> usize {
        frame_size + 1
    }

    #[inline]
    pub const fn saved_module_slot(self, frame_size: usize) -> usize {
        frame_size + 2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileConfig {
    pub tos_register_count: usize,
    pub hot_local_count: usize,
    pub frame_layout: FrameLayoutConfig,
}

impl CompileConfig {
    #[inline]
    pub const fn operand_stack_base(self, frame_size: usize) -> usize {
        self.frame_layout.operand_stack_base(frame_size)
    }

    #[inline]
    pub const fn operand_base_offset(self, frame_size: usize) -> usize {
        self.frame_layout.operand_base_offset(frame_size)
    }
}

pub const FAST_COMPILE_CONFIG: CompileConfig = CompileConfig {
    tos_register_count: 4,
    hot_local_count: MAX_HOT_LOCAL_COUNT,
    frame_layout: FrameLayoutConfig { metadata_slots: 3 },
};
