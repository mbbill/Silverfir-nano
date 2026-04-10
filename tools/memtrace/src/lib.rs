use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};
use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const MAX_FRAMES: usize = 32;
const STACK_SKIP_FRAMES: usize = 4;

const MODE_PENDING: u8 = 0;
const MODE_DISABLED: u8 = 1;
const MODE_ENABLED: u8 = 2;

thread_local! {
    static REENTRANT_DEPTH: Cell<u32> = const { Cell::new(0) };
}

static TRACE_MODE: AtomicU8 = AtomicU8::new(MODE_PENDING);
static TRACE_STATE: OnceLock<Mutex<TraceState>> = OnceLock::new();

unsafe extern "C" {
    fn backtrace(buffer: *mut *mut c_void, size: c_int) -> c_int;
}

#[derive(Clone, Debug)]
enum PendingEvent {
    Alloc {
        t_us: u64,
        ptr: usize,
        size: usize,
        stack_id: u32,
    },
    Free {
        t_us: u64,
        ptr: usize,
    },
    Realloc {
        t_us: u64,
        old_ptr: usize,
        new_ptr: usize,
        new_size: usize,
        stack_id: u32,
    },
    Exec {
        t_us: u64,
        base: usize,
        reserved: usize,
        used: usize,
    },
    ExecDrop {
        t_us: u64,
        base: usize,
    },
    Guard {
        t_us: u64,
        base: usize,
        reserved: usize,
        committed: usize,
    },
    GuardDrop {
        t_us: u64,
        base: usize,
    },
}

#[derive(Debug)]
struct TraceState {
    start: Instant,
    command_line: Vec<String>,
    output_path: Option<PathBuf>,
    writer: Option<BufWriter<File>>,
    stacks: Vec<Vec<usize>>,
    stack_lookup: HashMap<Vec<usize>, u32>,
    pending_events: Vec<PendingEvent>,
}

impl TraceState {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            command_line: Vec::new(),
            output_path: None,
            writer: None,
            stacks: Vec::new(),
            stack_lookup: HashMap::new(),
            pending_events: Vec::new(),
        }
    }

    fn timestamp_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    fn clear(&mut self) {
        self.command_line.clear();
        self.output_path = None;
        self.writer = None;
        self.stacks.clear();
        self.stack_lookup.clear();
        self.pending_events.clear();
    }

    fn ensure_writer(
        &mut self,
        command_line: Vec<String>,
        requested_path: Option<PathBuf>,
    ) -> std::io::Result<PathBuf> {
        if self.writer.is_some() {
            if !command_line.is_empty() {
                self.command_line = command_line;
            }
            return Ok(self
                .output_path
                .clone()
                .expect("memtrace output path initialized"));
        }

        if !command_line.is_empty() {
            self.command_line = command_line;
        }

        let path = requested_path.unwrap_or_else(default_trace_path);
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        write_meta_record(&mut writer, &self.command_line)?;
        for (stack_id, frames) in self.stacks.iter().enumerate() {
            write_stack_record(&mut writer, stack_id as u32, frames)?;
        }
        for event in &self.pending_events {
            write_event_record(&mut writer, event)?;
        }
        writer.flush()?;
        self.pending_events.clear();
        self.output_path = Some(path.clone());
        self.writer = Some(writer);
        Ok(path)
    }

    fn intern_stack(&mut self, frames: Vec<usize>) -> u32 {
        if let Some(existing) = self.stack_lookup.get(&frames).copied() {
            return existing;
        }
        let stack_id = self.stacks.len() as u32;
        self.stack_lookup.insert(frames.clone(), stack_id);
        self.stacks.push(frames);
        if let Some(writer) = self.writer.as_mut() {
            let frames = &self.stacks[stack_id as usize];
            let _ = write_stack_record(writer, stack_id, frames);
        }
        stack_id
    }

    fn record_event(&mut self, event: PendingEvent) {
        if let Some(writer) = self.writer.as_mut() {
            let _ = write_event_record(writer, &event);
        } else {
            self.pending_events.push(event);
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.flush()?;
        }
        Ok(())
    }
}

pub struct TrackingAllocator<A> {
    inner: A,
}

impl<A> TrackingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if tracing_disabled() || in_tracker() {
            return unsafe { self.inner.alloc(layout) };
        }
        with_tracker_disabled(|| {
            let ptr = unsafe { self.inner.alloc(layout) };
            if !ptr.is_null() {
                let stack_id = capture_stack_id();
                let mut state = trace_state().lock().unwrap();
                let t_us = state.timestamp_us();
                state.record_event(PendingEvent::Alloc {
                    t_us,
                    ptr: ptr as usize,
                    size: layout.size(),
                    stack_id,
                });
            }
            ptr
        })
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if tracing_disabled() || in_tracker() {
            unsafe { self.inner.dealloc(ptr, layout) };
            return;
        }
        with_tracker_disabled(|| {
            let mut state = trace_state().lock().unwrap();
            let t_us = state.timestamp_us();
            state.record_event(PendingEvent::Free {
                t_us,
                ptr: ptr as usize,
            });
            unsafe { self.inner.dealloc(ptr, layout) };
        });
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if tracing_disabled() || in_tracker() {
            return unsafe { self.inner.realloc(ptr, layout, new_size) };
        }
        with_tracker_disabled(|| {
            let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                let stack_id = capture_stack_id();
                let mut state = trace_state().lock().unwrap();
                let t_us = state.timestamp_us();
                state.record_event(PendingEvent::Realloc {
                    t_us,
                    old_ptr: ptr as usize,
                    new_ptr: new_ptr as usize,
                    new_size,
                    stack_id,
                });
            }
            new_ptr
        })
    }
}

pub fn configure_trace(
    command_line: Vec<String>,
    output_path: Option<PathBuf>,
) -> std::io::Result<PathBuf> {
    with_tracker_disabled(|| {
        let mut state = trace_state().lock().unwrap();
        let path = state.ensure_writer(command_line, output_path)?;
        TRACE_MODE.store(MODE_ENABLED, Ordering::Relaxed);
        Ok(path)
    })
}

pub fn disable_trace() {
    with_tracker_disabled(|| {
        let mut state = trace_state().lock().unwrap();
        state.clear();
        TRACE_MODE.store(MODE_DISABLED, Ordering::Relaxed);
    });
}

pub fn flush_trace() -> std::io::Result<()> {
    with_tracker_disabled(|| trace_state().lock().unwrap().flush())
}

pub fn trace_output_path() -> Option<PathBuf> {
    with_tracker_disabled(|| trace_state().lock().unwrap().output_path.clone())
}

pub fn record_exec_buffer_state(base: usize, reserved: usize, used: usize) {
    if tracing_disabled() {
        return;
    }
    with_tracker_disabled(|| {
        let mut state = trace_state().lock().unwrap();
        let t_us = state.timestamp_us();
        state.record_event(PendingEvent::Exec {
            t_us,
            base,
            reserved,
            used,
        });
    });
}

pub fn record_exec_buffer_drop(base: usize) {
    if tracing_disabled() {
        return;
    }
    with_tracker_disabled(|| {
        let mut state = trace_state().lock().unwrap();
        let t_us = state.timestamp_us();
        state.record_event(PendingEvent::ExecDrop {
            t_us,
            base,
        });
    });
}

pub fn record_guard_region_new(base: usize, reserved: usize, committed: usize) {
    if tracing_disabled() {
        return;
    }
    with_tracker_disabled(|| {
        let mut state = trace_state().lock().unwrap();
        let t_us = state.timestamp_us();
        state.record_event(PendingEvent::Guard {
            t_us,
            base,
            reserved,
            committed,
        });
    });
}

pub fn record_guard_region_grow(base: usize, committed: usize) {
    if tracing_disabled() {
        return;
    }
    with_tracker_disabled(|| {
        let mut state = trace_state().lock().unwrap();
        let t_us = state.timestamp_us();
        state.record_event(PendingEvent::Guard {
            t_us,
            base,
            reserved: 0,
            committed,
        });
    });
}

pub fn record_guard_region_drop(base: usize) {
    if tracing_disabled() {
        return;
    }
    with_tracker_disabled(|| {
        let mut state = trace_state().lock().unwrap();
        let t_us = state.timestamp_us();
        state.record_event(PendingEvent::GuardDrop {
            t_us,
            base,
        });
    });
}

fn trace_state() -> &'static Mutex<TraceState> {
    TRACE_STATE.get_or_init(|| Mutex::new(TraceState::new()))
}

fn capture_stack_id() -> u32 {
    let frames = capture_stack_frames();
    trace_state().lock().unwrap().intern_stack(frames)
}

fn capture_stack_frames() -> Vec<usize> {
    let mut frames = [std::ptr::null_mut::<c_void>(); MAX_FRAMES];
    let count = unsafe { backtrace(frames.as_mut_ptr(), MAX_FRAMES as c_int) };
    let mut out = Vec::new();
    if count <= 0 {
        return out;
    }
    for frame in frames
        .iter()
        .take(count as usize)
        .skip(STACK_SKIP_FRAMES)
        .copied()
    {
        if frame.is_null() {
            break;
        }
        out.push(frame as usize);
    }
    out
}

fn default_trace_path() -> PathBuf {
    std::env::temp_dir().join(format!("sf-memtrace-{}.jsonl", process::id()))
}

fn write_meta_record(writer: &mut BufWriter<File>, command_line: &[String]) -> std::io::Result<()> {
    write!(writer, "{{\"type\":\"meta\",\"version\":1,\"time_unit\":\"us\",\"command_line\":[")?;
    for (index, value) in command_line.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write_json_string(writer, value)?;
    }
    writeln!(writer, "]}}")
}

fn write_stack_record(
    writer: &mut BufWriter<File>,
    stack_id: u32,
    frames: &[usize],
) -> std::io::Result<()> {
    write!(writer, "{{\"type\":\"stack\",\"id\":{},\"frames\":[", stack_id)?;
    for (index, frame) in frames.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "\"0x{frame:016x}\"")?;
    }
    writeln!(writer, "]}}")
}

fn write_event_record(writer: &mut BufWriter<File>, event: &PendingEvent) -> std::io::Result<()> {
    match event {
        PendingEvent::Alloc {
            t_us,
            ptr,
            size,
            stack_id,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"alloc\",\"t_us\":{},\"ptr\":\"0x{:016x}\",\"size\":{},\"stack\":{}}}",
                t_us, ptr, size, stack_id
            )
        }
        PendingEvent::Free { t_us, ptr } => {
            writeln!(
                writer,
                "{{\"type\":\"free\",\"t_us\":{},\"ptr\":\"0x{:016x}\"}}",
                t_us, ptr
            )
        }
        PendingEvent::Realloc {
            t_us,
            old_ptr,
            new_ptr,
            new_size,
            stack_id,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"realloc\",\"t_us\":{},\"old_ptr\":\"0x{:016x}\",\"new_ptr\":\"0x{:016x}\",\"new_size\":{},\"stack\":{}}}",
                t_us, old_ptr, new_ptr, new_size, stack_id
            )
        }
        PendingEvent::Exec {
            t_us,
            base,
            reserved,
            used,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"exec\",\"t_us\":{},\"base\":\"0x{:016x}\",\"reserved\":{},\"used\":{}}}",
                t_us, base, reserved, used
            )
        }
        PendingEvent::ExecDrop { t_us, base } => {
            writeln!(
                writer,
                "{{\"type\":\"exec_drop\",\"t_us\":{},\"base\":\"0x{:016x}\"}}",
                t_us, base
            )
        }
        PendingEvent::Guard {
            t_us,
            base,
            reserved,
            committed,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"guard\",\"t_us\":{},\"base\":\"0x{:016x}\",\"reserved\":{},\"committed\":{}}}",
                t_us, base, reserved, committed
            )
        }
        PendingEvent::GuardDrop { t_us, base } => {
            writeln!(
                writer,
                "{{\"type\":\"guard_drop\",\"t_us\":{},\"base\":\"0x{:016x}\"}}",
                t_us, base
            )
        }
    }
}

fn write_json_string(writer: &mut BufWriter<File>, value: &str) -> std::io::Result<()> {
    write!(writer, "\"")?;
    for ch in value.chars() {
        match ch {
            '\\' => write!(writer, "\\\\")?,
            '"' => write!(writer, "\\\"")?,
            '\n' => write!(writer, "\\n")?,
            '\r' => write!(writer, "\\r")?,
            '\t' => write!(writer, "\\t")?,
            _ => write!(writer, "{ch}")?,
        }
    }
    write!(writer, "\"")
}

fn tracing_disabled() -> bool {
    TRACE_MODE.load(Ordering::Relaxed) == MODE_DISABLED
}

fn in_tracker() -> bool {
    REENTRANT_DEPTH.with(|depth| depth.get() != 0)
}

fn with_tracker_disabled<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            REENTRANT_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        }
    }
    REENTRANT_DEPTH.with(|depth| depth.set(depth.get() + 1));
    let _guard = Guard;
    f()
}
