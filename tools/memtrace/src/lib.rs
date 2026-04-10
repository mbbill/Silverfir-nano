use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
#[cfg(target_os = "macos")]
use std::ffi::{c_char, CStr};
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

#[cfg(target_os = "macos")]
const LC_SEGMENT_64: u32 = 0x19;

thread_local! {
    static REENTRANT_DEPTH: Cell<u32> = const { Cell::new(0) };
    static CURRENT_TAG_ID: Cell<u32> = const { Cell::new(0) };
}

static TRACE_MODE: AtomicU8 = AtomicU8::new(MODE_PENDING);
static TRACE_STATE: OnceLock<Mutex<TraceState>> = OnceLock::new();

unsafe extern "C" {
    fn backtrace(buffer: *mut *mut c_void, size: c_int) -> c_int;
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_header(image_index: u32) -> *const MachHeader64;
    fn _dyld_get_image_name(image_index: u32) -> *const c_char;
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct MachHeader64 {
    magic: u32,
    cputype: i32,
    cpusubtype: i32,
    filetype: u32,
    ncmds: u32,
    sizeofcmds: u32,
    flags: u32,
    reserved: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct LoadCommand {
    cmd: u32,
    cmdsize: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct SegmentCommand64 {
    cmd: u32,
    cmdsize: u32,
    segname: [u8; 16],
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    maxprot: i32,
    initprot: i32,
    nsects: u32,
    flags: u32,
}

#[derive(Clone, Debug)]
struct ImageInfo {
    base: usize,
    size: usize,
    path: String,
}

const UNTAGGED_TAG_NAME: &str = "untagged";

#[derive(Clone, Debug)]
enum PendingEvent {
    Alloc {
        t_us: u64,
        ptr: usize,
        size: usize,
        stack_id: u32,
        tag_id: u32,
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
        tag_id: u32,
    },
    Exec {
        t_us: u64,
        base: usize,
        reserved: usize,
        used: usize,
        tag_id: u32,
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
        tag_id: u32,
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
    tags: Vec<String>,
    tag_lookup: HashMap<String, u32>,
    stacks: Vec<Vec<usize>>,
    stack_lookup: HashMap<Vec<usize>, u32>,
    pending_events: Vec<PendingEvent>,
}

impl TraceState {
    fn new() -> Self {
        let mut tag_lookup = HashMap::new();
        tag_lookup.insert(UNTAGGED_TAG_NAME.to_string(), 0);
        Self {
            start: Instant::now(),
            command_line: Vec::new(),
            output_path: None,
            writer: None,
            tags: vec![UNTAGGED_TAG_NAME.to_string()],
            tag_lookup,
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
        self.tags.clear();
        self.tags.push(UNTAGGED_TAG_NAME.to_string());
        self.tag_lookup.clear();
        self.tag_lookup.insert(UNTAGGED_TAG_NAME.to_string(), 0);
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
        for image in loaded_images() {
            write_image_record(&mut writer, &image)?;
        }
        for (tag_id, name) in self.tags.iter().enumerate() {
            write_tag_record(&mut writer, tag_id as u32, name)?;
        }
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

    fn intern_tag(&mut self, name: &str) -> u32 {
        if let Some(existing) = self.tag_lookup.get(name).copied() {
            return existing;
        }
        let tag_id = self.tags.len() as u32;
        let owned = name.to_string();
        self.tag_lookup.insert(owned.clone(), tag_id);
        self.tags.push(owned);
        if let Some(writer) = self.writer.as_mut() {
            let _ = write_tag_record(writer, tag_id, &self.tags[tag_id as usize]);
        }
        tag_id
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

pub struct ScopeGuard {
    prev_tag_id: u32,
    active: bool,
}

impl<A> TrackingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if self.active {
            CURRENT_TAG_ID.with(|current| current.set(self.prev_tag_id));
        }
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
                let tag_id = current_tag_id();
                let mut state = trace_state().lock().unwrap();
                let t_us = state.timestamp_us();
                state.record_event(PendingEvent::Alloc {
                    t_us,
                    ptr: ptr as usize,
                    size: layout.size(),
                    stack_id,
                    tag_id,
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
                let tag_id = current_tag_id();
                let mut state = trace_state().lock().unwrap();
                let t_us = state.timestamp_us();
                state.record_event(PendingEvent::Realloc {
                    t_us,
                    old_ptr: ptr as usize,
                    new_ptr: new_ptr as usize,
                    new_size,
                    stack_id,
                    tag_id,
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
        CURRENT_TAG_ID.with(|current| current.set(0));
    });
}

pub fn flush_trace() -> std::io::Result<()> {
    with_tracker_disabled(|| trace_state().lock().unwrap().flush())
}

pub fn trace_output_path() -> Option<PathBuf> {
    with_tracker_disabled(|| trace_state().lock().unwrap().output_path.clone())
}

pub fn scope(name: &'static str) -> ScopeGuard {
    if tracing_disabled() || in_tracker() {
        return ScopeGuard {
            prev_tag_id: 0,
            active: false,
        };
    }
    with_tracker_disabled(|| {
        let tag_id = trace_state().lock().unwrap().intern_tag(name);
        let prev_tag_id = CURRENT_TAG_ID.with(|current| {
            let prev = current.get();
            current.set(tag_id);
            prev
        });
        ScopeGuard {
            prev_tag_id,
            active: true,
        }
    })
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
            tag_id: current_tag_id(),
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
            tag_id: current_tag_id(),
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
            tag_id: current_tag_id(),
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

fn current_tag_id() -> u32 {
    CURRENT_TAG_ID.with(|current| current.get())
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
    write!(writer, "{{\"type\":\"meta\",\"version\":3,\"time_unit\":\"us\",\"command_line\":[")?;
    for (index, value) in command_line.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write_json_string(writer, value)?;
    }
    writeln!(writer, "]}}")
}

fn write_tag_record(
    writer: &mut BufWriter<File>,
    tag_id: u32,
    name: &str,
) -> std::io::Result<()> {
    write!(writer, "{{\"type\":\"tag\",\"id\":{},\"name\":", tag_id)?;
    write_json_string(writer, name)?;
    writeln!(writer, "}}")
}

fn write_image_record(writer: &mut BufWriter<File>, image: &ImageInfo) -> std::io::Result<()> {
    write!(
        writer,
        "{{\"type\":\"image\",\"base\":\"0x{:016x}\",\"size\":{},\"path\":",
        image.base, image.size
    )?;
    write_json_string(writer, &image.path)?;
    writeln!(writer, "}}")
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
            tag_id,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"alloc\",\"t_us\":{},\"ptr\":\"0x{:016x}\",\"size\":{},\"stack\":{},\"tag\":{}}}",
                t_us, ptr, size, stack_id, tag_id
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
            tag_id,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"realloc\",\"t_us\":{},\"old_ptr\":\"0x{:016x}\",\"new_ptr\":\"0x{:016x}\",\"new_size\":{},\"stack\":{},\"tag\":{}}}",
                t_us, old_ptr, new_ptr, new_size, stack_id, tag_id
            )
        }
        PendingEvent::Exec {
            t_us,
            base,
            reserved,
            used,
            tag_id,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"exec\",\"t_us\":{},\"base\":\"0x{:016x}\",\"reserved\":{},\"used\":{},\"tag\":{}}}",
                t_us, base, reserved, used, tag_id
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
            tag_id,
        } => {
            writeln!(
                writer,
                "{{\"type\":\"guard\",\"t_us\":{},\"base\":\"0x{:016x}\",\"reserved\":{},\"committed\":{},\"tag\":{}}}",
                t_us, base, reserved, committed, tag_id
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

#[cfg(target_os = "macos")]
fn loaded_images() -> Vec<ImageInfo> {
    let mut images = Vec::new();
    let image_count = unsafe { _dyld_image_count() };
    for image_index in 0..image_count {
        let Some(image) = loaded_image(image_index) else {
            continue;
        };
        images.push(image);
    }
    images.sort_by_key(|image| image.base);
    images
}

#[cfg(not(target_os = "macos"))]
fn loaded_images() -> Vec<ImageInfo> {
    Vec::new()
}

#[cfg(target_os = "macos")]
fn loaded_image(image_index: u32) -> Option<ImageInfo> {
    let header = unsafe { _dyld_get_image_header(image_index) };
    if header.is_null() {
        return None;
    }
    let load_base = header as usize;
    let text_vmaddr = unsafe { text_segment_vmaddr(header)? };
    let image_end = unsafe { image_end_vmaddr(header)? };
    if image_end < text_vmaddr {
        return None;
    }
    let path_ptr = unsafe { _dyld_get_image_name(image_index) };
    let path = if path_ptr.is_null() {
        "<unknown>".to_string()
    } else {
        unsafe { CStr::from_ptr(path_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    Some(ImageInfo {
        base: load_base,
        size: (image_end - text_vmaddr) as usize,
        path,
    })
}

#[cfg(target_os = "macos")]
unsafe fn text_segment_vmaddr(header: *const MachHeader64) -> Option<u64> {
    let mut command = header.add(1) as *const LoadCommand;
    for _ in 0..(*header).ncmds {
        if (*command).cmd == LC_SEGMENT_64 {
            let segment = command as *const SegmentCommand64;
            let raw_name = &(*segment).segname;
            let end = raw_name.iter().position(|&b| b == 0).unwrap_or(raw_name.len());
            if &raw_name[..end] == b"__TEXT" {
                return Some((*segment).vmaddr);
            }
        }
        command = (command as *const u8).add((*command).cmdsize as usize) as *const LoadCommand;
    }
    None
}

#[cfg(target_os = "macos")]
unsafe fn image_end_vmaddr(header: *const MachHeader64) -> Option<u64> {
    let mut command = header.add(1) as *const LoadCommand;
    let mut max_end: Option<u64> = None;
    for _ in 0..(*header).ncmds {
        if (*command).cmd == LC_SEGMENT_64 {
            let segment = command as *const SegmentCommand64;
            let end = (*segment).vmaddr.saturating_add((*segment).vmsize);
            max_end = Some(max_end.map_or(end, |current| current.max(end)));
        }
        command = (command as *const u8).add((*command).cmdsize as usize) as *const LoadCommand;
    }
    max_end
}
