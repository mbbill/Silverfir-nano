use tracked_alloc::{Box, Rc, String, Vec};

#[test]
fn containers_are_standard_types_including_auto_traits() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Vec<u8>>();
    send_sync::<Box<u8>>();
    send_sync::<String>();
    let values: std::vec::Vec<u8> = Vec::from([1, 2]);
    let boxed: std::boxed::Box<u8> = Box::new(3);
    let text: std::string::String = String::from("text");
    let shared: std::rc::Rc<u8> = Rc::new(4);
    assert_eq!((values.len(), *boxed, text.len(), *shared), (2, 3, 4, 4));
}

#[test]
fn retype_keeps_allocation_and_checks_before_transfer() {
    let values = vec![1u32, 2, 3];
    let ptr = values.as_ptr().cast::<i32>();
    // SAFETY: every u32 bit pattern is a valid i32; no aliases survive.
    let values = unsafe { tracked_alloc::retype_vec::<u32, i32>(values) };
    assert_eq!(values.as_ptr(), ptr);
    assert_eq!(values, [1, 2, 3]);
    assert!(std::panic::catch_unwind(|| {
        // SAFETY: the checked layout rejection happens before any raw transfer.
        unsafe { tracked_alloc::retype_vec::<u8, u64>(vec![1u8]) }
    })
    .is_err());
    static DROPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    #[repr(transparent)]
    struct Word(u64);
    impl Drop for Word {
        fn drop(&mut self) {
            DROPS.fetch_add(self.0 as usize, std::sync::atomic::Ordering::SeqCst);
        }
    }
    assert!(std::panic::catch_unwind(|| {
        // SAFETY: layout matches, and the drop-glue check rejects the transfer.
        unsafe { tracked_alloc::retype_vec::<Word, u64>(vec![Word(1), Word(2)]) }
    })
    .is_err());
    assert_eq!(DROPS.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[cfg(feature = "memprof")]
#[test]
fn allocator_lifecycle_reset_failure_and_runtime_phases() {
    use std::alloc::{GlobalAlloc, Layout, System};
    use tracked_alloc::*;
    // Call the wrapper directly so test harness allocations cannot affect exact
    // totals. A separate executable test exercises it as the global allocator.
    let allocator = TrackingAllocator::new(System);
    reset_tracking();
    set_tracking_enabled(true);
    let phase = phase_span("test");
    let layout = Layout::from_size_align(64, 8).unwrap();
    let ptr = unsafe { allocator.alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    assert_eq!(unsafe { *ptr }, 0);
    assert_eq!(snapshot().total_bytes, 64);
    let ptr = unsafe { allocator.realloc(ptr, layout, 128) };
    assert!(!ptr.is_null());
    assert_eq!(snapshot().total_bytes, 128);
    unsafe { allocator.dealloc(ptr, Layout::from_size_align(128, 8).unwrap()) };
    drop(phase);
    let profile = profile();
    assert_eq!(profile.snapshot.total_bytes, 0);
    assert_eq!(profile.snapshot.peak_bytes, 128);
    assert_eq!(profile.snapshot.allocated_bytes, 128);
    assert_eq!(profile.snapshot.allocations, 1);
    assert_eq!(profile.snapshot.reallocations, 1);
    assert_eq!(profile.snapshot.deallocations, 1);
    assert_eq!(profile.phases[0].allocated_bytes, 128);

    struct FailingRealloc;
    unsafe impl GlobalAlloc for FailingRealloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, _ptr: *mut u8, _layout: Layout, _size: usize) -> *mut u8 {
            std::ptr::null_mut()
        }
    }
    let failing = TrackingAllocator::new(FailingRealloc);
    let ptr = unsafe { failing.alloc(layout) };
    assert!(!ptr.is_null());
    assert!(unsafe { failing.realloc(ptr, layout, 128) }.is_null());
    assert_eq!(
        snapshot().total_bytes,
        64,
        "failed realloc retains the old block"
    );
    set_tracking_enabled(false);
    unsafe { failing.dealloc(ptr, layout) };
    assert_eq!(
        snapshot().total_bytes,
        0,
        "paused tracking still releases recorded blocks"
    );
    set_tracking_enabled(true);
    let descriptor = AllocationDescriptor::new(RUNTIME_MEMORY_OWNER, RUNTIME_TYPE_CODE_BUFFER);
    let mut old = AllocationHandle::new(descriptor, AllocationState::new(4096));
    let old_phase = phase_span("before reset");
    assert_eq!(snapshot().code_buffer_bytes, 4096);
    reset_tracking();
    let current = AllocationHandle::new(descriptor, AllocationState::new(8192));
    old.update(AllocationState::new(100));
    drop(old);
    drop(old_phase);
    assert_eq!(snapshot().code_buffer_bytes, 8192);
    assert!(tracked_alloc::profile().phases.is_empty());
    drop(current);
    assert_eq!(snapshot().code_buffer_bytes, 0);
    assert_eq!(snapshot().peak_code_buffer_bytes, 8192);
    set_tracking_enabled(false);
    reset_tracking();
}
