#![no_std]

extern crate alloc;
#[cfg(feature = "tracking")]
extern crate std;

use boxed::Box;
#[cfg(feature = "tracking")]
use core::cmp::Ordering;
#[cfg(feature = "tracking")]
use core::fmt;
#[cfg(feature = "tracking")]
use core::hash::{Hash, Hasher};
#[cfg(feature = "tracking")]
use core::iter::FromIterator;
#[cfg(feature = "tracking")]
use core::marker::PhantomData;
#[cfg(feature = "tracking")]
use core::mem;
#[cfg(feature = "tracking")]
use core::ops::{Deref, DerefMut, RangeBounds};
#[cfg(feature = "tracking")]
use string::ToString;

#[cfg(feature = "tracking")]
use core::any::type_name;
#[cfg(feature = "tracking")]
use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
#[cfg(feature = "tracking")]
use std::backtrace::Backtrace;
#[cfg(feature = "tracking")]
use std::collections::BTreeMap as StdBTreeMap;
#[cfg(feature = "tracking")]
use std::sync::{Mutex, OnceLock};

mod inner {
    pub use alloc::vec::{Drain, IntoIter, Splice};
    pub type Vec<T> = alloc::vec::Vec<T>;
}

#[doc(hidden)]
pub mod __private {
    pub extern crate alloc as alloc_crate;
}

#[macro_export]
macro_rules! vec {
    ($($tt:tt)*) => {
        $crate::from_alloc_vec($crate::__private::alloc_crate::vec![$($tt)*])
    };
}

#[macro_export]
macro_rules! format {
    ($($tt:tt)*) => {
        $crate::__private::alloc_crate::format![$($tt)*]
    };
}

#[cfg(not(feature = "tracking"))]
pub type Vec<T> = inner::Vec<T>;

#[cfg(not(feature = "tracking"))]
#[inline]
pub fn from_alloc_vec<T>(inner: inner::Vec<T>) -> Vec<T> {
    inner
}

#[cfg(not(feature = "tracking"))]
#[inline]
pub fn into_alloc_vec<T>(value: Vec<T>) -> inner::Vec<T> {
    value
}

#[cfg(not(feature = "tracking"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub records: inner::Vec<VecSnapshot>,
    pub total_buffer_bytes: usize,
}

#[cfg(not(feature = "tracking"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VecSnapshot {
    pub id: u64,
    pub element_type: &'static str,
    pub element_size: usize,
    pub len: usize,
    pub capacity: usize,
    pub buffer_bytes: usize,
    pub buffer_ptr: usize,
    pub create_stack: Box<str>,
    pub last_capacity_change_stack: Option<Box<str>>,
}

#[cfg(not(feature = "tracking"))]
#[inline]
pub fn snapshot() -> RegistrySnapshot {
    RegistrySnapshot::default()
}

#[cfg(not(feature = "tracking"))]
#[inline]
pub fn reset_tracking() {}

#[cfg(feature = "tracking")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecSnapshot {
    pub id: u64,
    pub element_type: &'static str,
    pub element_size: usize,
    pub len: usize,
    pub capacity: usize,
    pub buffer_bytes: usize,
    pub buffer_ptr: usize,
    pub create_stack: Box<str>,
    pub last_capacity_change_stack: Option<Box<str>>,
}

#[cfg(feature = "tracking")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub records: inner::Vec<VecSnapshot>,
    pub total_buffer_bytes: usize,
}

#[cfg(feature = "tracking")]
#[derive(Debug)]
struct VecRecord {
    element_type: &'static str,
    element_size: usize,
    len: usize,
    capacity: usize,
    buffer_bytes: usize,
    buffer_ptr: usize,
    create_stack: Box<str>,
    last_capacity_change_stack: Option<Box<str>>,
}

#[cfg(feature = "tracking")]
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
#[cfg(feature = "tracking")]
static REGISTRY: OnceLock<Mutex<StdBTreeMap<u64, VecRecord>>> = OnceLock::new();

#[cfg(feature = "tracking")]
fn registry() -> &'static Mutex<StdBTreeMap<u64, VecRecord>> {
    REGISTRY.get_or_init(|| Mutex::new(StdBTreeMap::new()))
}

#[cfg(feature = "tracking")]
fn capture_stack() -> Box<str> {
    let text = Backtrace::force_capture().to_string();
    text.into_boxed_str()
}

#[cfg(feature = "tracking")]
fn buffer_ptr<T>(inner: &inner::Vec<T>) -> usize {
    if inner.capacity() == 0 {
        0
    } else {
        inner.as_ptr() as usize
    }
}

#[cfg(feature = "tracking")]
#[derive(Debug)]
struct TraceHandle {
    id: Option<u64>,
}

#[cfg(feature = "tracking")]
impl TraceHandle {
    fn new<T>(inner: &inner::Vec<T>) -> Self {
        let id = NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let record = VecRecord {
            element_type: type_name::<T>(),
            element_size: mem::size_of::<T>(),
            len: inner.len(),
            capacity: inner.capacity(),
            buffer_bytes: inner.capacity().saturating_mul(mem::size_of::<T>()),
            buffer_ptr: buffer_ptr(inner),
            create_stack: capture_stack(),
            last_capacity_change_stack: None,
        };
        registry().lock().unwrap().insert(id, record);
        Self { id: Some(id) }
    }

    fn sync<T>(&mut self, inner: &inner::Vec<T>) {
        let Some(id) = self.id else {
            return;
        };
        let mut registry = registry().lock().unwrap();
        let Some(record) = registry.get_mut(&id) else {
            return;
        };
        let new_capacity = inner.capacity();
        if record.capacity != new_capacity {
            record.last_capacity_change_stack = Some(capture_stack());
        }
        record.len = inner.len();
        record.capacity = new_capacity;
        record.buffer_bytes = new_capacity.saturating_mul(mem::size_of::<T>());
        record.buffer_ptr = buffer_ptr(inner);
    }

    fn remove(&mut self) {
        if let Some(id) = self.id.take() {
            registry().lock().unwrap().remove(&id);
        }
    }
}

#[cfg(feature = "tracking")]
impl Drop for TraceHandle {
    fn drop(&mut self) {
        self.remove();
    }
}

#[cfg(feature = "tracking")]
pub fn snapshot() -> RegistrySnapshot {
    let registry = registry().lock().unwrap();
    let mut records: inner::Vec<VecSnapshot> = registry
        .iter()
        .map(|(&id, record)| VecSnapshot {
            id,
            element_type: record.element_type,
            element_size: record.element_size,
            len: record.len,
            capacity: record.capacity,
            buffer_bytes: record.buffer_bytes,
            buffer_ptr: record.buffer_ptr,
            create_stack: record.create_stack.clone(),
            last_capacity_change_stack: record.last_capacity_change_stack.clone(),
        })
        .collect();
    records.sort_by(|a, b| {
        b.buffer_bytes
            .cmp(&a.buffer_bytes)
            .then_with(|| a.id.cmp(&b.id))
    });
    let total_buffer_bytes = records.iter().map(|record| record.buffer_bytes).sum();
    RegistrySnapshot {
        records,
        total_buffer_bytes,
    }
}

#[cfg(feature = "tracking")]
pub fn reset_tracking() {
    registry().lock().unwrap().clear();
    NEXT_ID.store(1, AtomicOrdering::Relaxed);
}

#[cfg(feature = "tracking")]
pub struct Vec<T> {
    inner: inner::Vec<T>,
    trace: TraceHandle,
}

#[cfg(feature = "tracking")]
impl<T> Vec<T> {
    #[inline]
    pub fn new() -> Self {
        Self::from_alloc_vec(inner::Vec::new())
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_alloc_vec(inner::Vec::with_capacity(capacity))
    }

    #[inline]
    pub fn from_alloc_vec(inner: inner::Vec<T>) -> Self {
        let trace = TraceHandle::new(&inner);
        Self { inner, trace }
    }

    #[inline]
    fn sync(&mut self) {
        self.trace.sync(&self.inner);
    }

    #[inline]
    fn mutate<R>(&mut self, f: impl FnOnce(&mut inner::Vec<T>) -> R) -> R {
        let result = f(&mut self.inner);
        self.sync();
        result
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.mutate(|inner| inner.push(value));
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.mutate(|inner| inner.pop())
    }

    #[inline]
    pub fn clear(&mut self) {
        self.mutate(inner::Vec::clear);
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.mutate(|inner| inner.truncate(len));
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve(additional));
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve_exact(additional));
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), collections::TryReserveError> {
        let result = self.inner.try_reserve(additional);
        self.sync();
        result
    }

    #[inline]
    pub fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), collections::TryReserveError> {
        let result = self.inner.try_reserve_exact(additional);
        self.sync();
        result
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.mutate(inner::Vec::shrink_to_fit);
    }

    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        self.mutate(|inner| inner.insert(index, element));
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        self.mutate(|inner| inner.remove(index))
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.mutate(|inner| inner.swap_remove(index))
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.inner.append(&mut other.inner);
        self.sync();
        other.sync();
    }

    #[inline]
    pub fn retain(&mut self, f: impl FnMut(&T) -> bool) {
        self.mutate(|inner| inner.retain(f));
    }

    #[inline]
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.mutate(inner::Vec::dedup);
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.mutate(|inner| inner.resize(new_len, value));
    }

    #[inline]
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        self.mutate(|inner| inner.resize_with(new_len, f));
    }

    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.mutate(|inner| inner.extend_from_slice(other));
    }

    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.mutate(|inner| inner.as_mut_slice().sort());
    }

    #[inline]
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.mutate(|inner| inner.sort_by(compare));
    }

    #[inline]
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.mutate(|inner| inner.as_mut_slice().sort_unstable());
    }

    #[inline]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let owner = self as *mut Self;
        let inner = self.inner.drain(range);
        Drain {
            inner: Some(inner),
            owner,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, T, I::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        let owner = self as *mut Self;
        let inner = self.inner.splice(range, replace_with);
        Splice {
            inner: Some(inner),
            owner,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        self.trace.remove();
        mem::take(&mut self.inner).into_boxed_slice()
    }

    #[inline]
    pub fn into_alloc_vec(mut self) -> inner::Vec<T> {
        self.trace.remove();
        mem::take(&mut self.inner)
    }
}

#[cfg(feature = "tracking")]
#[inline]
pub fn from_alloc_vec<T>(inner: inner::Vec<T>) -> Vec<T> {
    Vec::from_alloc_vec(inner)
}

#[cfg(feature = "tracking")]
#[inline]
pub fn into_alloc_vec<T>(value: Vec<T>) -> inner::Vec<T> {
    value.into_alloc_vec()
}

#[cfg(feature = "tracking")]
pub struct Drain<'a, T> {
    inner: Option<inner::Drain<'a, T>>,
    owner: *mut Vec<T>,
    marker: PhantomData<&'a mut Vec<T>>,
}

#[cfg(feature = "tracking")]
impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(Iterator::next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner
            .as_ref()
            .map(Iterator::size_hint)
            .unwrap_or((0, Some(0)))
    }
}

#[cfg(feature = "tracking")]
impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(DoubleEndedIterator::next_back)
    }
}

#[cfg(feature = "tracking")]
impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

#[cfg(feature = "tracking")]
impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        drop(self.inner.take());
        unsafe {
            (*self.owner).sync();
        }
    }
}

#[cfg(feature = "tracking")]
pub struct Splice<'a, T, I: Iterator<Item = T>> {
    inner: Option<inner::Splice<'a, I>>,
    owner: *mut Vec<T>,
    marker: PhantomData<&'a mut Vec<T>>,
}

#[cfg(feature = "tracking")]
impl<'a, T, I: Iterator<Item = T>> Iterator for Splice<'a, T, I> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(Iterator::next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner
            .as_ref()
            .map(Iterator::size_hint)
            .unwrap_or((0, Some(0)))
    }
}

#[cfg(feature = "tracking")]
impl<'a, T, I: Iterator<Item = T>> DoubleEndedIterator for Splice<'a, T, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(DoubleEndedIterator::next_back)
    }
}

#[cfg(feature = "tracking")]
impl<'a, T, I: Iterator<Item = T>> Drop for Splice<'a, T, I> {
    fn drop(&mut self) {
        drop(self.inner.take());
        unsafe {
            (*self.owner).sync();
        }
    }
}

#[cfg(feature = "tracking")]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        self.trace.remove();
    }
}

#[cfg(feature = "tracking")]
impl<T> Default for Vec<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "tracking")]
impl<T: Clone> Clone for Vec<T> {
    fn clone(&self) -> Self {
        from_alloc_vec(self.inner.clone())
    }
}

#[cfg(feature = "tracking")]
impl<T> From<inner::Vec<T>> for Vec<T> {
    fn from(value: inner::Vec<T>) -> Self {
        from_alloc_vec(value)
    }
}

#[cfg(feature = "tracking")]
impl<T> From<Vec<T>> for inner::Vec<T> {
    fn from(value: Vec<T>) -> Self {
        value.into_alloc_vec()
    }
}

#[cfg(feature = "tracking")]
impl<T: Clone> From<&[T]> for Vec<T> {
    fn from(value: &[T]) -> Self {
        from_alloc_vec(value.to_vec())
    }
}

#[cfg(feature = "tracking")]
impl<T, const N: usize> From<[T; N]> for Vec<T> {
    fn from(value: [T; N]) -> Self {
        from_alloc_vec(inner::Vec::from(value))
    }
}

#[cfg(feature = "tracking")]
impl<T> Extend<T> for Vec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter);
        self.sync();
    }
}

#[cfg(feature = "tracking")]
impl<'a, T: 'a + Clone> Extend<&'a T> for Vec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.inner.extend(iter.into_iter().cloned());
        self.sync();
    }
}

#[cfg(feature = "tracking")]
impl<T> FromIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        from_alloc_vec(inner::Vec::from_iter(iter))
    }
}

#[cfg(feature = "tracking")]
impl<T> Deref for Vec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

#[cfg(feature = "tracking")]
impl<T> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

#[cfg(feature = "tracking")]
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

#[cfg(feature = "tracking")]
impl<T> AsMut<[T]> for Vec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "tracking")]
impl<T> fmt::Debug for Vec<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "tracking")]
impl<T> PartialEq for Vec<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

#[cfg(feature = "tracking")]
impl<T> Eq for Vec<T> where T: Eq {}

#[cfg(feature = "tracking")]
impl<T> PartialOrd for Vec<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

#[cfg(feature = "tracking")]
impl<T> Ord for Vec<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

#[cfg(feature = "tracking")]
impl<T> Hash for Vec<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

#[cfg(feature = "tracking")]
impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = inner::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_alloc_vec().into_iter()
    }
}

#[cfg(feature = "tracking")]
impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(feature = "tracking")]
impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

pub mod boxed {
    pub use alloc::boxed::Box;
}

pub mod collections {
    pub use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap, TryReserveError, VecDeque};
}

pub mod rc {
    pub use alloc::rc::{Rc, Weak};
}

pub mod string {
    pub use alloc::string::{String, ToString};
}

pub mod vec {
    pub use crate::inner::IntoIter;

    pub use crate::{from_alloc_vec, into_alloc_vec, Vec};
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "tracking")]
    use std::sync::{Mutex, OnceLock};

    #[cfg(feature = "tracking")]
    fn tracking_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn vec_macro_matches_standard_behavior() {
        let values = vec![1u32, 2, 3];
        assert_eq!(values.len(), 3);
        assert_eq!(values[1], 2);
    }

    #[test]
    fn from_alloc_vec_round_trip() {
        let values = from_alloc_vec(alloc::vec![1u8, 2, 3]);
        let raw: inner::Vec<u8> = values.into();
        assert_eq!(raw, alloc::vec![1u8, 2, 3]);
    }

    #[test]
    fn alloc_like_surface_is_available() {
        #[cfg(feature = "tracking")]
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        #[cfg(feature = "tracking")]
        reset_tracking();

        let mut values = vec::Vec::new();
        values.push(1u32);
        let message = format!("values={}", values.len());
        let boxed = boxed::Box::new(message);
        let shared = rc::Rc::new(string::String::from(boxed.as_str()));
        let mut names = collections::BTreeSet::new();
        names.insert(shared.as_str());
        assert!(names.contains("values=1"));

        #[cfg(feature = "tracking")]
        {
            drop(names);
            drop(shared);
            drop(boxed);
            drop(values);
            assert!(snapshot().records.is_empty());
            reset_tracking();
        }
    }

    #[cfg(feature = "tracking")]
    #[test]
    fn tracking_records_lifecycle_and_clone() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_tracking();
        let mut values = Vec::<u32>::new();
        let snapshot0 = snapshot();
        assert_eq!(snapshot0.records.len(), 1);
        assert_eq!(snapshot0.records[0].element_type, type_name::<u32>());
        assert_eq!(snapshot0.records[0].len, 0);
        assert_eq!(snapshot0.records[0].capacity, 0);

        values.push(7);
        values.push(9);
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.records.len(), 1);
        assert_eq!(snapshot1.records[0].len, 2);
        assert!(snapshot1.records[0].buffer_bytes >= 2 * mem::size_of::<u32>());
        assert!(!snapshot1.records[0].create_stack.is_empty());

        let cloned = values.clone();
        let snapshot2 = snapshot();
        assert_eq!(
            snapshot2.records.len(),
            2,
            "ids={:?}",
            snapshot2
                .records
                .iter()
                .map(|record| record.id)
                .collect::<inner::Vec<_>>()
        );
        assert!(snapshot2.total_buffer_bytes >= snapshot1.total_buffer_bytes);

        drop(cloned);
        drop(values);
        assert!(snapshot().records.is_empty());
        reset_tracking();
    }

    #[cfg(feature = "tracking")]
    #[test]
    fn tracking_updates_after_drain_and_splice() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_tracking();
        let mut values = vec![1u8, 2, 3, 4];
        let drained: inner::Vec<u8> = values.drain(1..3).collect();
        assert_eq!(drained, alloc::vec![2u8, 3]);
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.records.len(), 1);
        assert_eq!(snapshot1.records[0].len, 2);

        let removed: inner::Vec<u8> = values.splice(1..1, alloc::vec![9u8, 10]).collect();
        assert!(removed.is_empty());
        let snapshot2 = snapshot();
        assert_eq!(snapshot2.records[0].len, 4);
        assert_eq!(&values[..], &[1, 9, 10, 4]);
        drop(values);
        reset_tracking();
    }
}
