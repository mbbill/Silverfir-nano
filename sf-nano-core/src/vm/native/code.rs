use crate::vm::native::ir::machine::MachineFunction;

/// Placeholder native code handle while the new machine backend is being built.
#[derive(Clone, Debug, Default)]
pub struct NativeCode {
    program: Option<MachineFunction>,
}

impl NativeCode {
    #[inline]
    pub const fn empty() -> Self {
        Self { program: None }
    }

    #[inline]
    pub fn from_program(program: MachineFunction) -> Self {
        Self {
            program: Some(program),
        }
    }

    #[inline]
    pub fn program(&self) -> Option<&MachineFunction> {
        self.program.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeCodeCache {
    compiled: bool,
}

impl NativeCodeCache {
    #[inline]
    pub const fn is_compiled(self) -> bool {
        self.compiled
    }

    #[inline]
    pub const fn compiled() -> Self {
        Self { compiled: true }
    }
}
