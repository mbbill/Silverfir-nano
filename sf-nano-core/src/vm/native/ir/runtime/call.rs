/// Shared local-call call-link layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MachineCallLinkLayout {
    /// Byte offset of the saved continuation pointer within call scratch.
    pub continuation_offset: i32,
    /// Byte offset of the saved caller frame pointer within call scratch.
    pub caller_frame_offset: i32,
    /// Byte offset of the saved caller result-base slot within call scratch.
    pub caller_result_base_offset: i32,
    /// Number of call-scratch slots reserved by the call-link prefix.
    pub slot_count: u16,
}
