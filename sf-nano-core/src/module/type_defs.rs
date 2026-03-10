//! WebAssembly 2.0 Type Definitions
//!
//! This module contains the function type definition used in the type section:
//! - Function types (0x60)

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::value_type::ValueType;

/// Function type - signature for callable functions
///
/// Defines the parameter types and result types for a function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    params: Vec<ValueType>,
    results: Vec<ValueType>,
}

impl FunctionType {
    /// Create a new function type
    pub fn new(params: Vec<ValueType>, results: Vec<ValueType>) -> Self {
        FunctionType { params, results }
    }

    /// Get the parameter types
    pub fn params(&self) -> &[ValueType] {
        &self.params
    }

    /// Get the result types
    pub fn results(&self) -> &[ValueType] {
        &self.results
    }
}

impl fmt::Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|v| {
                let mut buf = String::new();
                core::fmt::Write::write_fmt(&mut buf, format_args!("{}", v)).unwrap();
                buf
            })
            .collect();
        let results: Vec<String> = self
            .results
            .iter()
            .map(|v| {
                let mut buf = String::new();
                core::fmt::Write::write_fmt(&mut buf, format_args!("{}", v)).unwrap();
                buf
            })
            .collect();
        write!(f, "({}) -> ({})", params.join(", "), results.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_function_type() {
        let ft = FunctionType::new(vec![ValueType::I32, ValueType::I64], vec![ValueType::F32]);

        assert_eq!(ft.params().len(), 2);
        assert_eq!(ft.results().len(), 1);
    }
}
