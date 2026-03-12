use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use core::fmt;

use crate::utils::{limits::LimitsError, payload::PayloadError};

/// Macro for creating WasmError with lazy formatting to minimize stack usage.
#[macro_export]
macro_rules! wasm_error {
    (malformed, $($arg:tt)*) => {
        $crate::error::WasmError::malformed_fmt(|| alloc::format!($($arg)*))
    };
    (invalid, $($arg:tt)*) => {
        $crate::error::WasmError::invalid_fmt(|| alloc::format!($($arg)*))
    };
    (unlinkable, $($arg:tt)*) => {
        $crate::error::WasmError::unlinkable_fmt(|| alloc::format!($($arg)*))
    };
    (exhaustion, $($arg:tt)*) => {
        $crate::error::WasmError::exhaustion_fmt(|| alloc::format!($($arg)*))
    };
    (trap, $($arg:tt)*) => {
        $crate::error::WasmError::trap_fmt(|| alloc::format!($($arg)*))
    };
    (exit, $($arg:tt)*) => {
        $crate::error::WasmError::exit_fmt(|| alloc::format!($($arg)*))
    };
    (internal, $($arg:tt)*) => {
        $crate::error::WasmError::internal_fmt(|| alloc::format!($($arg)*))
    };
}

#[derive(Debug)]
enum WasmErrorInner {
    Malformed { message: String },
    Invalid { message: String },
    Unlinkable { message: String },
    Exhaustion { message: String },
    Trap { message: String },
    Exit { code: i32 },
    Internal { message: String },
}

/// Public error type - a thin pointer wrapper (8 bytes) for efficient stack usage.
#[derive(Debug)]
pub struct WasmError {
    inner: Box<WasmErrorInner>,
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.inner {
            WasmErrorInner::Malformed { message, .. } => write!(f, "Malformed: {}", message),
            WasmErrorInner::Invalid { message, .. } => write!(f, "Invalid: {}", message),
            WasmErrorInner::Unlinkable { message, .. } => write!(f, "Unlinkable: {}", message),
            WasmErrorInner::Exhaustion { message, .. } => write!(f, "Exhaustion: {}", message),
            WasmErrorInner::Trap { message, .. } => write!(f, "Trap: {}", message),
            WasmErrorInner::Exit { code, .. } => {
                write!(f, "Exit: Process exited with code {}", code)
            }
            WasmErrorInner::Internal { message, .. } => write!(f, "Internal error: {}", message),
        }
    }
}

impl Clone for WasmError {
    fn clone(&self) -> Self {
        match &*self.inner {
            WasmErrorInner::Malformed { message, .. } => Self::malformed(message.clone()),
            WasmErrorInner::Invalid { message, .. } => Self::invalid(message.clone()),
            WasmErrorInner::Unlinkable { message, .. } => Self::unlinkable(message.clone()),
            WasmErrorInner::Exhaustion { message, .. } => Self::exhaustion(message.clone()),
            WasmErrorInner::Trap { message, .. } => Self::trap(message.clone()),
            WasmErrorInner::Exit { code, .. } => Self::exit_with_code(*code),
            WasmErrorInner::Internal { message, .. } => Self::internal(message.clone()),
        }
    }
}

impl WasmError {
    #[cold]
    #[inline(never)]
    pub fn malformed(message: String) -> Self {
        Self {
            inner: Box::new(WasmErrorInner::Malformed { message }),
        }
    }

    #[cold]
    #[inline(never)]
    pub fn malformed_fmt(f: impl FnOnce() -> String) -> Self {
        Self::malformed(f())
    }

    #[cold]
    #[inline(never)]
    pub fn invalid(message: String) -> Self {
        Self {
            inner: Box::new(WasmErrorInner::Invalid { message }),
        }
    }

    #[cold]
    #[inline(never)]
    pub fn invalid_fmt(f: impl FnOnce() -> String) -> Self {
        Self::invalid(f())
    }

    #[cold]
    #[inline(never)]
    pub fn unlinkable(message: String) -> Self {
        Self {
            inner: Box::new(WasmErrorInner::Unlinkable { message }),
        }
    }

    #[cold]
    #[inline(never)]
    pub fn unlinkable_fmt(f: impl FnOnce() -> String) -> Self {
        Self::unlinkable(f())
    }

    #[cold]
    #[inline(never)]
    pub fn exhaustion(message: String) -> Self {
        Self {
            inner: Box::new(WasmErrorInner::Exhaustion { message }),
        }
    }

    #[cold]
    #[inline(never)]
    pub fn exhaustion_fmt(f: impl FnOnce() -> String) -> Self {
        Self::exhaustion(f())
    }

    #[cold]
    #[inline(never)]
    pub fn trap(message: String) -> Self {
        Self {
            inner: Box::new(WasmErrorInner::Trap { message }),
        }
    }

    #[cold]
    #[inline(never)]
    pub fn trap_fmt(f: impl FnOnce() -> String) -> Self {
        Self::trap(f())
    }

    #[cold]
    #[inline(never)]
    pub fn exit_with_code(code: i32) -> Self {
        Self {
            inner: Box::new(WasmErrorInner::Exit { code }),
        }
    }

    #[cold]
    #[inline(never)]
    pub fn exit(message: String) -> Self {
        let code = message
            .strip_prefix("Process exited with code ")
            .and_then(|s| s.trim().parse::<i32>().ok())
            .unwrap_or(1);
        Self::exit_with_code(code)
    }

    #[cold]
    #[inline(never)]
    pub fn exit_fmt(f: impl FnOnce() -> String) -> Self {
        Self::exit(f())
    }

    #[cold]
    #[inline(never)]
    pub fn internal(message: String) -> Self {
        Self {
            inner: Box::new(WasmErrorInner::Internal { message }),
        }
    }

    #[cold]
    #[inline(never)]
    pub fn internal_fmt(f: impl FnOnce() -> String) -> Self {
        Self::internal(f())
    }

    pub fn is_malformed(&self) -> bool {
        matches!(&*self.inner, WasmErrorInner::Malformed { .. })
    }

    pub fn is_trap(&self) -> bool {
        matches!(&*self.inner, WasmErrorInner::Trap { .. })
    }

    pub fn is_unlinkable(&self) -> bool {
        matches!(&*self.inner, WasmErrorInner::Unlinkable { .. })
    }

    pub fn is_exit(&self) -> bool {
        matches!(&*self.inner, WasmErrorInner::Exit { .. })
    }

    pub fn message(&self) -> String {
        match &*self.inner {
            WasmErrorInner::Malformed { message, .. }
            | WasmErrorInner::Invalid { message, .. }
            | WasmErrorInner::Unlinkable { message, .. }
            | WasmErrorInner::Exhaustion { message, .. }
            | WasmErrorInner::Trap { message, .. }
            | WasmErrorInner::Internal { message, .. } => message.clone(),
            WasmErrorInner::Exit { code, .. } => format!("Process exited with code {}", code),
        }
    }

    pub fn class(&self) -> &'static str {
        match &*self.inner {
            WasmErrorInner::Malformed { .. } => "malformed",
            WasmErrorInner::Invalid { .. } => "invalid",
            WasmErrorInner::Unlinkable { .. } => "unlinkable",
            WasmErrorInner::Exhaustion { .. } => "exhaustion",
            WasmErrorInner::Trap { .. } => "trap",
            WasmErrorInner::Exit { .. } => "exit",
            WasmErrorInner::Internal { .. } => "internal",
        }
    }

    pub fn exit_code(&self) -> Option<i32> {
        match &*self.inner {
            WasmErrorInner::Exit { code, .. } => Some(*code),
            _ => None,
        }
    }
}

impl From<PayloadError> for WasmError {
    fn from(error: PayloadError) -> Self {
        WasmError::malformed(format!("{}", error))
    }
}

impl From<LimitsError> for WasmError {
    fn from(error: LimitsError) -> Self {
        WasmError::invalid(format!("Limits error: {}", error))
    }
}
