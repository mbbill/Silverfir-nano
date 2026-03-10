//! Internal interpreter stack for WebAssembly execution.
//!
//! High-performance stack using a simple owned `Vec<RawValue>` buffer.
//! No thread-local storage — the stack is a plain struct suitable for `no_std`.

use alloc::vec;
use alloc::vec::Vec;

use super::raw_value::{
    as_f32, as_f64, as_i32, as_i64, from_f32, from_f64, from_i32, from_i64, RawValue,
};
use crate::value_type::ValueType;

/// High-performance interpreter stack using pre-allocated buffer.
/// Designed for minimal overhead in tight interpreter loops.
#[derive(Debug)]
pub struct InterpreterStack {
    /// Pre-allocated buffer to avoid heap allocations during execution
    buffer: Vec<RawValue>,
    /// Current stack pointer (index into buffer)
    sp: usize,
}

impl InterpreterStack {
    /// Create new stack with default capacity
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create stack with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            sp: 0,
        }
    }

    /// Create stack with exact pre-computed capacity from function validation.
    /// Pre-allocates the exact amount of stack space needed.
    pub fn with_exact_capacity(total_capacity: usize) -> Self {
        Self {
            buffer: vec![0; total_capacity],
            sp: 0,
        }
    }

    // -- Push / Pop / Peek ---------------------------------------------------

    /// Push raw value onto stack
    #[inline(always)]
    pub fn push(&mut self, value: RawValue) {
        debug_assert!(
            self.sp < self.buffer.len(),
            "InterpreterStack overflow: ensure_capacity/with_exact_capacity must guarantee space"
        );
        unsafe {
            *self.buffer.get_unchecked_mut(self.sp) = value;
        }
        self.sp += 1;
    }

    /// Pop raw value from stack
    #[inline(always)]
    pub fn pop(&mut self) -> RawValue {
        debug_assert!(self.sp > 0, "Stack underflow");
        self.sp -= 1;
        unsafe { *self.buffer.get_unchecked(self.sp) }
    }

    /// Peek top of stack without popping
    #[inline(always)]
    pub fn peek(&self) -> RawValue {
        debug_assert!(self.sp > 0, "Stack underflow");
        unsafe { *self.buffer.get_unchecked(self.sp - 1) }
    }

    /// Peek at a specific index from the bottom of the stack
    #[inline(always)]
    pub fn peek_at_index(&self, index: usize) -> RawValue {
        debug_assert!(index < self.sp, "Stack index out of bounds");
        unsafe { *self.buffer.get_unchecked(index) }
    }

    /// Get stack length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.sp
    }

    /// Pop multiple values — returns slice of values in stack order
    #[inline(always)]
    pub fn pop_n(&mut self, n: usize) -> &[RawValue] {
        debug_assert!(self.sp >= n, "Stack underflow");
        self.sp -= n;
        unsafe { self.buffer.get_unchecked(self.sp..self.sp + n) }
    }

    // -- Locals ---------------------------------------------------------------

    /// Get local variable by offset from base
    #[inline(always)]
    pub fn get_local(&self, base: usize, index: usize) -> RawValue {
        debug_assert!(base + index < self.sp, "Local access out of bounds");
        unsafe { *self.buffer.get_unchecked(base + index) }
    }

    /// Set local variable by offset from base
    #[inline(always)]
    pub fn set_local(&mut self, base: usize, index: usize, value: RawValue) {
        debug_assert!(base + index < self.sp, "Local access out of bounds");
        unsafe {
            *self.buffer.get_unchecked_mut(base + index) = value;
        }
    }

    /// Initialize locals with default zero values in bulk.
    /// WebAssembly defaults are zero for all numeric types and null for refs,
    /// both represented as 0 in RawValue storage.
    pub fn push_locals(&mut self, value_types: &[ValueType]) {
        let count = value_types.len();
        if count == 0 {
            return;
        }
        let new_sp = self.sp + count;
        if self.buffer.len() < new_sp {
            self.buffer.resize(new_sp, 0);
        }
        self.buffer[self.sp..new_sp].fill(0);
        self.sp = new_sp;
    }

    // -- Capacity -------------------------------------------------------------

    /// Ensure the stack has at least the specified capacity
    pub fn ensure_capacity(&mut self, required_capacity: usize) {
        if self.buffer.len() < required_capacity {
            self.buffer.resize(required_capacity, 0);
        }
    }

    // -- Reduce (in-place binary ops) -----------------------------------------

    /// Reduce top two i32 values to one result in-place: res = f(lhs, rhs)
    #[inline(always)]
    pub fn reduce_i32<F: FnOnce(i32, i32) -> i32>(&mut self, f: F) {
        debug_assert!(self.sp >= 2, "Stack underflow on reduce_i32");
        unsafe {
            let rhs = as_i32(*self.buffer.get_unchecked(self.sp - 1));
            let lhs_i = self.sp - 2;
            let lhs = as_i32(*self.buffer.get_unchecked(lhs_i));
            *self.buffer.get_unchecked_mut(lhs_i) = from_i32(f(lhs, rhs));
            self.sp -= 1;
        }
    }

    /// Reduce top two i64 values to one result in-place
    #[inline(always)]
    pub fn reduce_i64<F: FnOnce(i64, i64) -> i64>(&mut self, f: F) {
        debug_assert!(self.sp >= 2, "Stack underflow on reduce_i64");
        unsafe {
            let rhs = as_i64(*self.buffer.get_unchecked(self.sp - 1));
            let lhs_i = self.sp - 2;
            let lhs = as_i64(*self.buffer.get_unchecked(lhs_i));
            *self.buffer.get_unchecked_mut(lhs_i) = from_i64(f(lhs, rhs));
            self.sp -= 1;
        }
    }

    /// Reduce top two f32 values to one result in-place
    #[inline(always)]
    pub fn reduce_f32<F: FnOnce(f32, f32) -> f32>(&mut self, f: F) {
        debug_assert!(self.sp >= 2, "Stack underflow on reduce_f32");
        unsafe {
            let rhs = as_f32(*self.buffer.get_unchecked(self.sp - 1));
            let lhs_i = self.sp - 2;
            let lhs = as_f32(*self.buffer.get_unchecked(lhs_i));
            *self.buffer.get_unchecked_mut(lhs_i) = from_f32(f(lhs, rhs));
            self.sp -= 1;
        }
    }

    /// Reduce top two f64 values to one result in-place
    #[inline(always)]
    pub fn reduce_f64<F: FnOnce(f64, f64) -> f64>(&mut self, f: F) {
        debug_assert!(self.sp >= 2, "Stack underflow on reduce_f64");
        unsafe {
            let rhs = as_f64(*self.buffer.get_unchecked(self.sp - 1));
            let lhs_i = self.sp - 2;
            let lhs = as_f64(*self.buffer.get_unchecked(lhs_i));
            *self.buffer.get_unchecked_mut(lhs_i) = from_f64(f(lhs, rhs));
            self.sp -= 1;
        }
    }

    // -- Unary ops (in-place) -------------------------------------------------

    /// Apply a unary i32 operation to the top of stack in-place
    #[inline(always)]
    pub fn unop_i32<F: FnOnce(i32) -> i32>(&mut self, f: F) {
        debug_assert!(self.sp >= 1, "Stack underflow on unop_i32");
        unsafe {
            let top_i = self.sp - 1;
            let v = as_i32(*self.buffer.get_unchecked(top_i));
            *self.buffer.get_unchecked_mut(top_i) = from_i32(f(v));
        }
    }

    /// Apply a unary i64 operation to the top of stack in-place
    #[inline(always)]
    pub fn unop_i64<F: FnOnce(i64) -> i64>(&mut self, f: F) {
        debug_assert!(self.sp >= 1, "Stack underflow on unop_i64");
        unsafe {
            let top_i = self.sp - 1;
            let v = as_i64(*self.buffer.get_unchecked(top_i));
            *self.buffer.get_unchecked_mut(top_i) = from_i64(f(v));
        }
    }

    // -- Control flow helpers -------------------------------------------------

    /// Shift results for function returns.
    /// Moves the top `top_n` values down to `local_start`, discarding locals.
    pub fn shift_results(&mut self, top_n: usize, local_start: usize) {
        if local_start >= self.sp {
            return;
        }

        if top_n == 0 {
            self.sp = local_start;
            return;
        }

        let end = self.sp;
        let start = end - top_n;
        if local_start != start {
            self.buffer.copy_within(start..end, local_start);
        }
        self.sp = local_start + top_n;
    }
}

impl Default for InterpreterStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_capacity_constructor() {
        let stack = InterpreterStack::with_exact_capacity(8);
        assert_eq!(stack.len(), 0);
        assert_eq!(stack.buffer.len(), 8);
    }

    #[test]
    fn test_push_pop() {
        let mut stack = InterpreterStack::with_exact_capacity(10);
        stack.push(42);
        stack.push(84);
        assert_eq!(stack.len(), 2);
        assert_eq!(stack.pop(), 84);
        assert_eq!(stack.pop(), 42);
        assert_eq!(stack.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Stack underflow")]
    fn test_pop_underflow() {
        let mut stack = InterpreterStack::new();
        stack.pop();
    }
}
