/// Native runtime context placeholder.
///
/// The real runtime/executor will be rebuilt around MachineIR. This placeholder
/// exists only so shared debug code can continue to compile while native
/// execution is being rewired.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeContext;
