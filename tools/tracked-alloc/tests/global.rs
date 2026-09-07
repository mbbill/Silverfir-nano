#![cfg(feature = "memprof")]
#[global_allocator]
static ALLOCATOR: tracked_alloc::TrackingAllocator<std::alloc::System> =
    tracked_alloc::TrackingAllocator::new(std::alloc::System);

#[test]
fn standard_allocations_are_measured_without_profiler_recursion() {
    tracked_alloc::reset_tracking();
    tracked_alloc::set_tracking_enabled(true);
    let data = std::hint::black_box(vec![7u8; 65536]);
    let live = tracked_alloc::snapshot();
    assert!(live.total_bytes >= data.capacity());
    let again = tracked_alloc::snapshot();
    assert_eq!(
        again.total_bytes, live.total_bytes,
        "snapshots exclude their own allocations"
    );
    drop(data);
    assert_eq!(
        tracked_alloc::snapshot().total_bytes,
        live.total_bytes - 65536
    );
    // Cross-thread frees and concurrent reallocations must not deadlock or
    // overwrite records for a just-reused address.
    let workers: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..100 {
                    let mut data = vec![1u8; 128];
                    data.resize(1024, 2);
                    std::hint::black_box(data);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    tracked_alloc::set_tracking_enabled(false);
    assert!(tracked_alloc::profile().snapshot.peak_bytes >= 65536);
    tracked_alloc::reset_tracking();
}
