use core::fmt;

use tracked_alloc::string::String;

use crate::collections;
use crate::utils::{leb128::ReadError as Leb128ReadError, payload::PayloadError};
use crate::vm::tag::TagIdentity;
use crate::vm::value::Value;
use crate::RefValue;

/// An error from parsing, validation, linking, execution or a host callback.
///
/// The representation is private. Use the classification/accessor methods;
/// host callbacks can return [`Self::trap`] or [`crate::Caller::throw`].
#[derive(Debug, Clone, PartialEq)]
pub struct WasmError {
    pub(crate) repr: ErrorRepr,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ErrorRepr {
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
    /// Host-side throw inbound channel produced by `Caller::throw`.
    /// This variant is VM-internal — the runtime-call entry
    /// consumes it and converts it into `NativeCallStatus::Thrown`. It
    /// should never reach the embedder.
    HostThrowValues {
        tag: TagIdentity,
        args: collections::Vec<crate::Value>,
    },
    HostThrow {
        tag: TagIdentity,
        args: collections::Vec<Value>,
    },
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.repr {
            ErrorRepr::Malformed(message) => write!(f, "Malformed: {}", message),
            ErrorRepr::Invalid(message) => write!(f, "Invalid: {}", message),
            ErrorRepr::Unlinkable(message) => write!(f, "Unlinkable: {}", message),
            ErrorRepr::Exhaustion(message) => write!(f, "Exhaustion: {}", message),
            ErrorRepr::Trap(message) => write!(f, "Trap: {}", message),
            ErrorRepr::Exit(code) => write!(f, "Exit: Process exited with code {}", code),
            ErrorRepr::Internal(message) => write!(f, "Internal error: {}", message),
            ErrorRepr::Exception {
                module_tag_name, ..
            } => match module_tag_name {
                Some(name) => write!(f, "Uncaught exception: {}", name),
                None => write!(f, "Uncaught exception"),
            },
            ErrorRepr::HostThrow { .. } | ErrorRepr::HostThrowValues { .. } => {
                write!(f, "Host throw (internal)")
            }
        }
    }
}

impl WasmError {
    #[cold]
    #[inline(never)]
    pub const fn malformed(message: &'static str) -> Self {
        Self {
            repr: ErrorRepr::Malformed(message),
        }
    }

    #[cold]
    #[inline(never)]
    pub const fn invalid(message: &'static str) -> Self {
        Self {
            repr: ErrorRepr::Invalid(message),
        }
    }

    #[cold]
    #[inline(never)]
    pub const fn unlinkable(message: &'static str) -> Self {
        Self {
            repr: ErrorRepr::Unlinkable(message),
        }
    }

    #[cold]
    #[inline(never)]
    pub const fn exhaustion(message: &'static str) -> Self {
        Self {
            repr: ErrorRepr::Exhaustion(message),
        }
    }

    #[cold]
    #[inline(never)]
    pub const fn trap(message: &'static str) -> Self {
        Self {
            repr: ErrorRepr::Trap(message),
        }
    }

    #[cold]
    #[inline(never)]
    pub const fn exit_with_code(code: i32) -> Self {
        Self {
            repr: ErrorRepr::Exit(code),
        }
    }

    #[cold]
    #[inline(never)]
    pub const fn internal(message: &'static str) -> Self {
        Self {
            repr: ErrorRepr::Internal(message),
        }
    }

    /// Constructed at instantiation / `memory.grow` when a module would
    /// reach more Wasm memory pages than
    /// the engine's `wasm_memory_max_pages` allows. Unlinkable per
    /// the spec's taxonomy — the module is valid, it just cannot be
    /// instantiated or grown in this configuration.
    #[cold]
    #[inline(never)]
    pub(crate) const fn memory_exceeds_runtime_limit() -> Self {
        Self {
            repr: ErrorRepr::Unlinkable(
                "memory exceeds runtime configured limit (wasm_memory_max_pages)",
            ),
        }
    }

    pub const fn is_malformed(&self) -> bool {
        matches!(&self.repr, ErrorRepr::Malformed(_))
    }

    pub const fn is_trap(&self) -> bool {
        matches!(&self.repr, ErrorRepr::Trap(_))
    }

    pub const fn is_unlinkable(&self) -> bool {
        matches!(&self.repr, ErrorRepr::Unlinkable(_))
    }

    pub const fn is_exit(&self) -> bool {
        matches!(&self.repr, ErrorRepr::Exit(_))
    }

    pub const fn message(&self) -> &'static str {
        match &self.repr {
            ErrorRepr::Malformed(message)
            | ErrorRepr::Invalid(message)
            | ErrorRepr::Unlinkable(message)
            | ErrorRepr::Exhaustion(message)
            | ErrorRepr::Trap(message)
            | ErrorRepr::Internal(message) => message,
            ErrorRepr::Exit(_) => "Process exited with code",
            ErrorRepr::Exception { .. } => "uncaught wasm exception",
            ErrorRepr::HostThrow { .. } | ErrorRepr::HostThrowValues { .. } => {
                "host throw (internal)"
            }
        }
    }

    pub const fn class(&self) -> &'static str {
        match &self.repr {
            ErrorRepr::Malformed(_) => "malformed",
            ErrorRepr::Invalid(_) => "invalid",
            ErrorRepr::Unlinkable(_) => "unlinkable",
            ErrorRepr::Exhaustion(_) => "exhaustion",
            ErrorRepr::Trap(_) => "trap",
            ErrorRepr::Exit(_) => "exit",
            ErrorRepr::Internal(_) => "internal",
            ErrorRepr::Exception { .. } => "exception",
            ErrorRepr::HostThrow { .. } | ErrorRepr::HostThrowValues { .. } => "host_throw",
        }
    }

    #[inline]
    pub const fn is_exception(&self) -> bool {
        matches!(&self.repr, ErrorRepr::Exception { .. })
    }

    pub const fn exit_code(&self) -> Option<i32> {
        match &self.repr {
            ErrorRepr::Exit(code) => Some(*code),
            _ => None,
        }
    }
}

impl core::error::Error for WasmError {}

impl WasmError {
    /// Reference to an uncaught exception; resolve its payload through the instance.
    pub const fn exception(&self) -> Option<RefValue> {
        match &self.repr {
            ErrorRepr::Exception { exn, .. } => Some(*exn),
            _ => None,
        }
    }

    /// Identity of the tag attached to an uncaught exception.
    pub const fn exception_tag(&self) -> Option<TagIdentity> {
        match &self.repr {
            ErrorRepr::Exception { tag, .. } => Some(*tag),
            _ => None,
        }
    }

    /// Export name associated with the uncaught exception tag, when available.
    pub fn exception_tag_name(&self) -> Option<&str> {
        match &self.repr {
            ErrorRepr::Exception {
                module_tag_name, ..
            } => module_tag_name.as_deref(),
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
