use sf_nano_core::wasi::{set_wasi_ctx, wasi_imports, WasiContextBuilder};
use sf_nano_core::{set_backend_mode, BackendMode, Instance, Value};

use std::env;
use std::fs;
use std::path::Path;
use std::process;

fn parse_value(text: &str) -> Result<Value, String> {
    let (kind, raw) = text
        .split_once(':')
        .ok_or_else(|| format!("argument '{text}' must be typed as kind:value"))?;
    match kind {
        "i32" => Ok(Value::I32(parse_i64(raw)? as i32)),
        "i64" => Ok(Value::I64(parse_i64(raw)?)),
        "f32" => Ok(Value::F32(raw.parse::<f32>().map_err(|e| e.to_string())?)),
        "f64" => Ok(Value::F64(raw.parse::<f64>().map_err(|e| e.to_string())?)),
        "f32bits" => Ok(Value::F32(f32::from_bits(parse_u64(raw)? as u32))),
        "f64bits" => Ok(Value::F64(f64::from_bits(parse_u64(raw)?))),
        other => Err(format!("unsupported value kind '{other}'")),
    }
}

fn parse_i64(text: &str) -> Result<i64, String> {
    if let Some(hex) = text.strip_prefix("-0x") {
        i64::from_str_radix(hex, 16)
            .map(|v| -v)
            .map_err(|e| e.to_string())
    } else if let Some(hex) = text.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
            .map(|v| v as i64)
            .map_err(|e| e.to_string())
    } else {
        text.parse::<i64>().map_err(|e| e.to_string())
    }
}

fn parse_u64(text: &str) -> Result<u64, String> {
    if let Some(hex) = text.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        text.parse::<u64>().map_err(|e| e.to_string())
    }
}

fn print_value(value: &Value) {
    match value {
        Value::I32(v) => println!("i32:{v} bits=0x{:08x}", *v as u32),
        Value::I64(v) => println!("i64:{v} bits=0x{:016x}", *v as u64),
        Value::F32(v) => println!("f32:{v:?} bits=0x{:08x}", v.to_bits()),
        Value::F64(v) => println!("f64:{v:?} bits=0x{:016x}", v.to_bits()),
        other => println!("{other:?}"),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: cargo run -p sf-nano-core --example invoke_export -- <wasm> <export> [typed-args...] [--dump-memory <offset> <len>]"
    );
    process::exit(2);
}

fn main() {
    let mut args = env::args().skip(1);
    let wasm_path = args.next().unwrap_or_else(|| usage());
    let export = args.next().unwrap_or_else(|| usage());

    let mut dump_memory: Option<(usize, usize)> = None;
    let mut values = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--dump-memory" {
            let offset = args
                .next()
                .unwrap_or_else(|| usage())
                .parse::<usize>()
                .unwrap_or_else(|_| usage());
            let len = args
                .next()
                .unwrap_or_else(|| usage())
                .parse::<usize>()
                .unwrap_or_else(|_| usage());
            dump_memory = Some((offset, len));
            continue;
        }
        values.push(parse_value(&arg).unwrap_or_else(|err| {
            eprintln!("error: {err}");
            process::exit(2);
        }));
    }

    set_backend_mode(BackendMode::Native);

    let data = fs::read(&wasm_path).unwrap_or_else(|err| {
        eprintln!("error reading '{}': {err}", wasm_path);
        process::exit(1);
    });
    let module_name = Path::new(&wasm_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("module")
        .to_string();

    let ctx = WasiContextBuilder::new()
        .args(&[module_name])
        .preopen_dir(".", Path::new("."))
        .inherit_env()
        .build();
    set_wasi_ctx(ctx);

    let imports = wasi_imports();
    let mut instance = Instance::new(&data, &imports).unwrap_or_else(|err| {
        eprintln!("instantiation failed: {err}");
        process::exit(1);
    });

    let results = instance.invoke(&export, &values).unwrap_or_else(|err| {
        eprintln!("invoke failed: {err}");
        process::exit(1);
    });

    println!("result_count={}", results.len());
    for value in &results {
        print_value(value);
    }

    if let Some((offset, len)) = dump_memory {
        let memory = instance.memory().unwrap_or_else(|| {
            eprintln!("memory dump requested but module has no memory");
            process::exit(1);
        });
        let end = offset
            .checked_add(len)
            .filter(|end| *end <= memory.len())
            .unwrap_or_else(|| {
                eprintln!(
                    "memory dump out of range: offset={} len={} memory_len={}",
                    offset,
                    len,
                    memory.len()
                );
                process::exit(1);
            });
        for (index, byte) in memory[offset..end].iter().enumerate() {
            println!("mem[{}]=0x{:02x}", offset + index, byte);
        }
    }
}
