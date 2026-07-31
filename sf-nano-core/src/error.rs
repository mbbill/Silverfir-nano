use core::fmt;

use tracked_alloc::string::String;

use crate::collections;
use crate::utils::{
    leb128::ReadError as Leb128ReadError, limits::LimitsError, payload::PayloadError,
};
use crate::vm::tag::TagIdentity;
use crate::vm::value::{RefValue, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum WasmError {
    Malformed(&'static str),
    Invalid(&'static str),
    Unlinkable(&'static str),
    Exhaustion(&'static str),
    Trap(&'static str),
    Exit(i32),
    Internal(&'static str),
    /// Uncaught wasm exception surfaced to the embedder. Produced by EH
    /// helpers when a throw propagates past every active `try_table` handler
    /// in the current invocation.
    Exception {
        exn: RefValue,
        tag: TagIdentity,
        module_tag_name: Option<String>,
    },
    /// Host-side throw inbound channel. A host callback returns
    /// `Err(WasmError::HostThrow { .. })` to signal a wasm-catchable
    /// exception. This variant is VM-internal — the runtime-call entry
    /// consumes it and converts it into `NativeCallStatus::Thrown`. It
    /// should never reach the embedder.
    HostThrow {
        tag: TagIdentity,
        args: collections::Vec<Value>,
    },
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmError::Malformed(message) => write!(f, "Malformed: {}", message),
            WasmError::Invalid(message) => write!(f, "Invalid: {}", message),
            WasmError::Unlinkable(message) => write!(f, "Unlinkable: {}", message),
            WasmError::Exhaustion(message) => write!(f, "Exhaustion: {}", message),
            WasmError::Trap(message) => write!(f, "Trap: {}", message),
            WasmError::Exit(code) => write!(f, "Exit: Process exited with code {}", code),
            WasmError::Internal(message) => write!(f, "Internal error: {}", message),
            WasmError::Exception {
                module_tag_name, ..
            } => match module_tag_name {
                Some(name) => write!(f, "Uncaught exception: {}", name),
                None => write!(f, "Uncaught exception"),
            },
            WasmError::HostThrow { .. } => write!(f, "Host throw (internal)"),
        }
    }
}

impl WasmError {
    #[cold]
    #[inline(never)]
    pub const fn malformed(message: &'static str) -> Self {
        Self::Malformed(message)
    }

    #[cold]
    #[inline(never)]
    pub const fn invalid(message: &'static str) -> Self {
        Self::Invalid(message)
    }

    #[cold]
    #[inline(never)]
    pub const fn unlinkable(message: &'static str) -> Self {
        Self::Unlinkable(message)
    }

    #[cold]
    #[inline(never)]
    pub const fn exhaustion(message: &'static str) -> Self {
        Self::Exhaustion(message)
    }

    #[cold]
    #[inline(never)]
    pub const fn trap(message: &'static str) -> Self {
        Self::Trap(message)
    }

    #[cold]
    #[inline(never)]
    pub const fn exit_with_code(code: i32) -> Self {
        Self::Exit(code)
    }

    #[cold]
    #[inline(never)]
    pub const fn internal(message: &'static str) -> Self {
        Self::Internal(message)
    }

    /// Constructed at instantiation / `memory.grow` when a module would
    /// reach more Wasm memory pages than
    /// the engine's `wasm_memory_max_pages` allows. Unlinkable per
    /// the spec's taxonomy — the module is valid, it just cannot be
    /// instantiated or grown in this configuration.
    #[cold]
    #[inline(never)]
    pub const fn memory_exceeds_runtime_limit() -> Self {
        Self::Unlinkable("memory exceeds runtime configured limit (wasm_memory_max_pages)")
    }

    pub const fn is_malformed(&self) -> bool {
        matches!(self, WasmError::Malformed(_))
    }

    pub const fn is_trap(&self) -> bool {
        matches!(self, WasmError::Trap(_))
    }

    pub const fn is_unlinkable(&self) -> bool {
        matches!(self, WasmError::Unlinkable(_))
    }

    pub const fn is_exit(&self) -> bool {
        matches!(self, WasmError::Exit(_))
    }

    pub const fn message(&self) -> &'static str {
        match self {
            WasmError::Malformed(message)
            | WasmError::Invalid(message)
            | WasmError::Unlinkable(message)
            | WasmError::Exhaustion(message)
            | WasmError::Trap(message)
            | WasmError::Internal(message) => message,
            WasmError::Exit(_) => "Process exited with code",
            WasmError::Exception { .. } => "uncaught wasm exception",
            WasmError::HostThrow { .. } => "host throw (internal)",
        }
    }

    pub const fn class(&self) -> &'static str {
        match self {
            WasmError::Malformed(_) => "malformed",
            WasmError::Invalid(_) => "invalid",
            WasmError::Unlinkable(_) => "unlinkable",
            WasmError::Exhaustion(_) => "exhaustion",
            WasmError::Trap(_) => "trap",
            WasmError::Exit(_) => "exit",
            WasmError::Internal(_) => "internal",
            WasmError::Exception { .. } => "exception",
            WasmError::HostThrow { .. } => "host_throw",
        }
    }

    #[inline]
    pub const fn is_exception(&self) -> bool {
        matches!(self, WasmError::Exception { .. })
    }

    pub const fn exit_code(&self) -> Option<i32> {
        match self {
            WasmError::Exit(code) => Some(*code),
            _ => None,
        }
    }
}

impl From<PayloadError> for WasmError {
    fn from(error: PayloadError) -> Self {
        match error {
            PayloadError::UnexpectedEndOfInput(_) => {
                WasmError::malformed("unexpected end of input")
            }
            PayloadError::InvalidData(_) => WasmError::malformed("invalid payload data"),
            PayloadError::InvalidLEB128(leb) => match leb {
                Leb128ReadError::InsufficientData => {
                    WasmError::malformed("unexpected end of input")
                }
                Leb128ReadError::ValueTooLong => WasmError::malformed("invalid LEB128 value"),
                Leb128ReadError::UnusedBitsSet => WasmError::malformed("invalid LEB128 value"),
            },
            PayloadError::RewindOutOfBounds(_) => {
                WasmError::internal("payload rewind out of bounds")
            }
        }
    }
}

impl From<LimitsError> for WasmError {
    fn from(error: LimitsError) -> Self {
        match error {
            LimitsError::MinLargerThanMax => WasmError::invalid("min larger than max"),
            LimitsError::MaxLargerThanDefaultMax => {
                WasmError::invalid("max larger than default max")
            }
            LimitsError::MinLargerThanDefaultMax => {
                WasmError::invalid("min larger than default max")
            }
        }
    }
}
