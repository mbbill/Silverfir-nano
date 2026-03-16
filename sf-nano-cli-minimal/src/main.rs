#![no_std]
#![no_main]

extern crate alloc;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write;
use sf_nano_core::{active_backend, Import, Instance, Value};

// — Minimal allocator using libc malloc/free —

struct LibcAllocator;

unsafe impl alloc::alloc::GlobalAlloc for LibcAllocator {
    unsafe fn alloc(&self, layout: alloc::alloc::Layout) -> *mut u8 {
        unsafe { libc::malloc(layout.size()) as *mut u8 }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: alloc::alloc::Layout) {
        unsafe { libc::free(ptr as *mut libc::c_void) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, _layout: alloc::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe { libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8 }
    }
}

#[global_allocator]
static ALLOC: LibcAllocator = LibcAllocator;

// — Stderr writer —

struct Stderr;

impl Write for Stderr {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe { libc::write(2, s.as_ptr() as *const libc::c_void, s.len()) };
        Ok(())
    }
}

// — Stdout writer —

struct Stdout;

impl Write for Stdout {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe { libc::write(1, s.as_ptr() as *const libc::c_void, s.len()) };
        Ok(())
    }
}

macro_rules! eprintln {
    ($($arg:tt)*) => {{
        let _ = writeln!(Stderr, $($arg)*);
    }};
}

macro_rules! println {
    ($($arg:tt)*) => {{
        let _ = writeln!(Stdout, $($arg)*);
    }};
}

// — Panic handler —

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        let msg = b"panic\n";
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
        libc::exit(101)
    }
}

#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

// — File reading via libc —

fn read_file(path: &[u8]) -> Option<Vec<u8>> {
    unsafe {
        let fd = libc::open(path.as_ptr() as *const libc::c_char, libc::O_RDONLY);
        if fd < 0 {
            return None;
        }

        // Get file size
        let end = libc::lseek(fd, 0, libc::SEEK_END);
        if end < 0 {
            libc::close(fd);
            return None;
        }
        libc::lseek(fd, 0, libc::SEEK_SET);

        let size = end as usize;
        let mut buf = vec![0u8; size];
        let mut read_total = 0;
        while read_total < size {
            let n = libc::read(fd, buf[read_total..].as_mut_ptr() as *mut libc::c_void, size - read_total);
            if n <= 0 {
                libc::close(fd);
                return None;
            }
            read_total += n as usize;
        }
        libc::close(fd);
        Some(buf)
    }
}

// — Embedded self-test wasm modules —

// (module (func (export "test") (result i32) i32.const 42))
static WASM_CONST: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60,
    0x00, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x08, 0x01, 0x04, 0x74,
    0x65, 0x73, 0x74, 0x00, 0x00, 0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a,
    0x0b,
];

// (module (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))
static WASM_ADD: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
    0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01,
    0x03, 0x61, 0x64, 0x64, 0x00, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x20,
    0x00, 0x20, 0x01, 0x6a, 0x0b,
];

// (module (func (export "sum_to") (param i32) (result i32) ... loop sum 1..n ...))
static WASM_LOOP: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60,
    0x01, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x0a, 0x01, 0x06,
    0x73, 0x75, 0x6d, 0x5f, 0x74, 0x6f, 0x00, 0x00, 0x0a, 0x25, 0x01, 0x23,
    0x01, 0x02, 0x7f, 0x02, 0x40, 0x03, 0x40, 0x20, 0x01, 0x20, 0x00, 0x4e,
    0x0d, 0x01, 0x20, 0x01, 0x41, 0x01, 0x6a, 0x21, 0x01, 0x20, 0x02, 0x20,
    0x01, 0x6a, 0x21, 0x02, 0x0c, 0x00, 0x0b, 0x0b, 0x20, 0x02, 0x0b,
];

// (module (func $fib (export "fib") (param i32) (result i32) ... recursive fib ...))
static WASM_FIB: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60,
    0x01, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03,
    0x66, 0x69, 0x62, 0x00, 0x00, 0x0a, 0x1e, 0x01, 0x1c, 0x00, 0x20, 0x00,
    0x41, 0x01, 0x4c, 0x04, 0x7f, 0x20, 0x00, 0x05, 0x20, 0x00, 0x41, 0x02,
    0x6b, 0x10, 0x00, 0x20, 0x00, 0x41, 0x01, 0x6b, 0x10, 0x00, 0x6a, 0x0b,
    0x0b,
];

// (module
//   (import "env" "print_i32" (func $print (param i32)))
//   (func (export "run")
//     i32.const 'H' call $print  i32.const 'e' call $print
//     i32.const 'l' call $print  i32.const 'l' call $print
//     i32.const 'o' call $print  i32.const '\n' call $print))
static WASM_HOSTCALL: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x60,
    0x01, 0x7f, 0x00, 0x60, 0x00, 0x00, 0x02, 0x11, 0x01, 0x03, 0x65, 0x6e,
    0x76, 0x09, 0x70, 0x72, 0x69, 0x6e, 0x74, 0x5f, 0x69, 0x33, 0x32, 0x00,
    0x00, 0x03, 0x02, 0x01, 0x01, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e,
    0x00, 0x01, 0x0a, 0x21, 0x01, 0x1f, 0x00, 0x41, 0xc8, 0x00, 0x10, 0x00,
    0x41, 0xe5, 0x00, 0x10, 0x00, 0x41, 0xec, 0x00, 0x10, 0x00, 0x41, 0xec,
    0x00, 0x10, 0x00, 0x41, 0xef, 0x00, 0x10, 0x00, 0x41, 0x0a, 0x10, 0x00,
    0x0b,
];

fn host_print_i32(
    _caller: &mut sf_nano_core::vm::entities::Caller,
    args: &[Value],
    _results: &mut [Value],
) -> Result<(), sf_nano_core::WasmError> {
    if let Some(Value::I32(ch)) = args.first() {
        let byte = *ch as u8;
        unsafe { libc::write(1, &byte as *const u8 as *const libc::c_void, 1) };
    }
    Ok(())
}

fn check_i32(results: &[Value], expected: i32) -> bool {
    matches!(results.first(), Some(Value::I32(v)) if *v == expected)
}

fn run_self_test() -> i32 {
    // Print active backend
    match active_backend() {
        Ok(b) => println!("Backend: {}", b.as_str()),
        Err(e) => { eprintln!("No backend: {}", e); return 1; }
    }

    let empty_imports = vec![];
    let mut passed = 0u32;
    let mut failed = 0u32;

    // Test 1: const 42
    {
        let mut inst = match Instance::new(WASM_CONST, &empty_imports) {
            Ok(i) => i,
            Err(e) => { eprintln!("FAIL const: instantiate: {}", e); return 1; }
        };
        match inst.invoke("test", &[]) {
            Ok(ref r) if check_i32(r, 42) => { println!("PASS const: test() = 42"); passed += 1; }
            Ok(ref r) => { eprintln!("FAIL const: expected 42, got {:?}", r); failed += 1; }
            Err(e) => { eprintln!("FAIL const: {}", e); failed += 1; }
        }
    }

    // Test 2: add(13, 29) = 42
    {
        let mut inst = match Instance::new(WASM_ADD, &empty_imports) {
            Ok(i) => i,
            Err(e) => { eprintln!("FAIL add: instantiate: {}", e); return 1; }
        };
        let args = [Value::I32(13), Value::I32(29)];
        match inst.invoke("add", &args) {
            Ok(ref r) if check_i32(r, 42) => { println!("PASS add: add(13, 29) = 42"); passed += 1; }
            Ok(ref r) => { eprintln!("FAIL add: expected 42, got {:?}", r); failed += 1; }
            Err(e) => { eprintln!("FAIL add: {}", e); failed += 1; }
        }
    }

    // Test 3: sum_to(10) = 55  (loop, locals, branches)
    {
        let mut inst = match Instance::new(WASM_LOOP, &empty_imports) {
            Ok(i) => i,
            Err(e) => { eprintln!("FAIL loop: instantiate: {}", e); return 1; }
        };
        let args = [Value::I32(10)];
        match inst.invoke("sum_to", &args) {
            Ok(ref r) if check_i32(r, 55) => { println!("PASS loop: sum_to(10) = 55"); passed += 1; }
            Ok(ref r) => { eprintln!("FAIL loop: expected 55, got {:?}", r); failed += 1; }
            Err(e) => { eprintln!("FAIL loop: {}", e); failed += 1; }
        }
    }

    // Test 4: fib(10) = 55  (recursion, if/else)
    {
        let mut inst = match Instance::new(WASM_FIB, &empty_imports) {
            Ok(i) => i,
            Err(e) => { eprintln!("FAIL fib: instantiate: {}", e); return 1; }
        };
        let args = [Value::I32(10)];
        match inst.invoke("fib", &args) {
            Ok(ref r) if check_i32(r, 55) => { println!("PASS fib: fib(10) = 55"); passed += 1; }
            Ok(ref r) => { eprintln!("FAIL fib: expected 55, got {:?}", r); failed += 1; }
            Err(e) => { eprintln!("FAIL fib: {}", e); failed += 1; }
        }
    }

    // Test 5: host function import — wasm calls host print_i32 to print "Hello"
    {
        let imports = vec![
            Import::func("env", "print_i32", host_print_i32),
        ];
        let mut inst = match Instance::new(WASM_HOSTCALL, &imports) {
            Ok(i) => i,
            Err(e) => { eprintln!("FAIL hostcall: instantiate: {}", e); return 1; }
        };
        match inst.invoke("run", &[]) {
            Ok(_) => { println!("PASS hostcall: wasm printed \"Hello\" via host import"); passed += 1; }
            Err(e) => { eprintln!("FAIL hostcall: {}", e); failed += 1; }
        }
    }

    println!("{} passed, {} failed", passed, failed);
    if failed > 0 { 1 } else { 0 }
}

// — Entry point —

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    if argc < 2 {
        return run_self_test();
    }

    // Check for --self-test flag
    let arg1_ptr = unsafe { *argv.offset(1) };
    let arg1 = unsafe {
        let len = libc::strlen(arg1_ptr as *const libc::c_char);
        core::slice::from_raw_parts(arg1_ptr, len)
    };
    if arg1 == b"--self-test" {
        return run_self_test();
    }

    // Get argv[1] as a null-terminated path
    let path = unsafe {
        let len = libc::strlen(arg1_ptr as *const libc::c_char);
        core::slice::from_raw_parts(arg1_ptr, len + 1) // include null terminator for open()
    };

    let data = match read_file(path) {
        Some(d) => d,
        None => {
            eprintln!("Error reading wasm file");
            return 1;
        }
    };

    let imports = vec![];
    let mut instance = match Instance::new(&data, &imports) {
        Ok(inst) => inst,
        Err(err) => {
            eprintln!("Error instantiating module: {}", err);
            return 1;
        }
    };

    let result = instance.invoke("_start", &[]);
    let result = match result {
        Err(ref err) if err.to_string().contains("not found") => {
            instance.invoke("main", &[])
        }
        _ => result,
    };

    match result {
        Ok(_) => 0,
        Err(err) => {
            if let Some(code) = err.exit_code() {
                return code;
            }
            eprintln!("Error: {}", err);
            1
        }
    }
}
