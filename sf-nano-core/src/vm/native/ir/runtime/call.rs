/// Shared local-call call-link layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MachineCallLinkLayout {
    pub continuation_offset: i32,
    pub caller_frame_offset: i32,
}
