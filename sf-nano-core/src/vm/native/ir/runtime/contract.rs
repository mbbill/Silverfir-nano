use super::MachineCallLinkLayout;

/// Shared runtime contract for machine IR execution.
///
/// This contains only execution-boundary details that remain after VM/runtime
/// semantics have been lowered above machine IR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineRuntimeContract {
    pub call_link: MachineCallLinkLayout,
}
