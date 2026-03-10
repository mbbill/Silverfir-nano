//! Internal interpreter stack for WebAssembly execution.
//!
//! High-performance stack using a simple owned `Vec<RawValue>` buffer.
//! No thread-local storage.

use alloc::vec;
use alloc::vec::Vec;

use super::raw_value::{
    as_f32, as_f64, as_i32, as_i64, from_f32, from_f64, from_i32, from_i64, RawValue,
};
use crate::value_type::ValueType;

#[derive(Debug)]
pub struct InterpreterStack {
    buffer: Vec<RawValue>,
    sp: usize,
}

impl InterpreterStack {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            sp: 0,
        }
    }

    pub fn with_exact_capacity(total_capacity: usize) -> Self {
        Self {
            buffer: vec![0; total_capacity],
            sp: 0,
        }
    }

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

    #[inline(always)]
    pub fn pop(&mut self) -> RawValue {
        debug_assert!(self.sp > 0, "Stack underflow");
        self.sp -= 1;
        unsafe { *self.buffer.get_unchecked(self.sp) }
    }

    #[inline(always)]
    pub fn peek(&self) -> RawValue {
        debug_assert!(self.sp > 0, "Stack underflow");
        unsafe { *self.buffer.get_unchecked(self.sp - 1) }
    }

    #[inline(always)]
    pub fn peek_at_index(&self, index: usize) -> RawValue {
        debug_assert!(index < self.sp, "Stack index out of bounds");
        unsafe { *self.buffer.get_unchecked(index) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.sp
    }

    #[inline(always)]
    pub fn pop_n(&mut self, n: usize) -> &[RawValue] {
        debug_assert!(self.sp >= n, "Stack underflow");
        self.sp -= n;
        unsafe { self.buffer.get_unchecked(self.sp..self.sp + n) }
    }

    #[inline(always)]
    pub fn get_local(&self, base: usize, index: usize) -> RawValue {
        debug_assert!(base + index < self.sp, "Local access out of bounds");
        unsafe { *self.buffer.get_unchecked(base + index) }
    }

    #[inline(always)]
    pub fn set_local(&mut self, base: usize, index: usize, value: RawValue) {
        debug_assert!(base + index < self.sp, "Local access out of bounds");
        unsafe {
            *self.buffer.get_unchecked_mut(base + index) = value;
        }
    }

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

    pub fn ensure_capacity(&mut self, required_capacity: usize) {
        if self.buffer.len() < required_capacity {
            self.buffer.resize(required_capacity, 0);
        }
    }

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

    #[inline(always)]
    pub fn unop_i32<F: FnOnce(i32) -> i32>(&mut self, f: F) {
        debug_assert!(self.sp >= 1, "Stack underflow on unop_i32");
        unsafe {
            let top_i = self.sp - 1;
            let v = as_i32(*self.buffer.get_unchecked(top_i));
            *self.buffer.get_unchecked_mut(top_i) = from_i32(f(v));
        }
    }

    #[inline(always)]
    pub fn unop_i64<F: FnOnce(i64) -> i64>(&mut self, f: F) {
        debug_assert!(self.sp >= 1, "Stack underflow on unop_i64");
        unsafe {
            let top_i = self.sp - 1;
            let v = as_i64(*self.buffer.get_unchecked(top_i));
            *self.buffer.get_unchecked_mut(top_i) = from_i64(f(v));
        }
    }

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
