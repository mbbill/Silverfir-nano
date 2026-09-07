#![cfg(feature = "wasi")]

use sf_nano_core::wasi::{wasi_imports, WasiContextBuilder};
use sf_nano_core::{Config, Engine, Import, Instance, Tier, Value};
use std::cell::RefCell;
use std::rc::Rc;

fn probe_module(namespace: &str) -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
        (import "{namespace}" "args_sizes_get" (func $args (param i32 i32) (result i32)))
        (import "{namespace}" "args_get" (func $argv (param i32 i32) (result i32)))
        (import "{namespace}" "environ_sizes_get" (func $env (param i32 i32) (result i32)))
        (import "{namespace}" "fd_close" (func $close (param i32) (result i32)))
        (import "{namespace}" "fd_fdstat_get" (func $stat (param i32 i32) (result i32)))
        (memory 1)
        (func (export "argc") (result i32)
            i32.const 0 i32.const 4 call $args drop i32.const 0 i32.load)
        (func (export "first_arg") (result i32)
            i32.const 8 i32.const 32 call $argv drop i32.const 32 i32.load8_u)
        (func (export "envc") (result i32)
            i32.const 0 i32.const 4 call $env drop i32.const 0 i32.load)
        (func (export "close_stdout") (result i32) i32.const 1 call $close)
        (func (export "stdout_status") (result i32) i32.const 1 i32.const 64 call $stat))"#
    ))
    .unwrap()
}

fn result(instance: &mut Instance, name: &str) -> i32 {
    match instance.invoke(name, &[]).unwrap().as_slice() {
        [Value::I32(value)] => *value,
        values => panic!("unexpected {name} result: {values:?}"),
    }
}

#[test]
fn distinct_import_sets_isolate_arguments_environment_and_descriptors() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let first_imports = wasi_imports(
            WasiContextBuilder::new()
                .args(["alpha"])
                .env("A", "1")
                .build(),
        );
        let second_imports =
            wasi_imports(WasiContextBuilder::new().args(["beta", "extra"]).build());
        let wasm = probe_module("wasi_snapshot_preview1");
        let mut first = Instance::new(&engine, &wasm, &first_imports).unwrap();
        let mut second = Instance::new(&engine, &wasm, &second_imports).unwrap();
        // Legacy and preview1 names in the same import set share one context.
        let mut shared =
            Instance::new(&engine, &probe_module("wasi_unstable"), &first_imports).unwrap();
        drop(first_imports);
        drop(second_imports);
        for _ in 0..3 {
            assert_eq!(result(&mut first, "argc"), 1);
            assert_eq!(result(&mut second, "argc"), 2);
            assert_eq!(result(&mut first, "first_arg"), b'a' as i32);
            assert_eq!(result(&mut second, "first_arg"), b'b' as i32);
            assert_eq!(result(&mut first, "envc"), 1);
            assert_eq!(result(&mut second, "envc"), 0);
        }
        assert_eq!(result(&mut first, "close_stdout"), 0);
        assert_ne!(result(&mut first, "stdout_status"), 0);
        assert_ne!(result(&mut shared, "stdout_status"), 0);
        assert_eq!(result(&mut second, "stdout_status"), 0);
        drop(first);
        assert_eq!(result(&mut shared, "argc"), 1);
        assert_eq!(result(&mut second, "argc"), 2);
    }
}

#[test]
fn nested_calls_keep_the_calling_instances_context() {
    let outer_wasm = wat::parse_str(
        r#"(module
        (import "host" "nested" (func $nested (result i32)))
        (import "wasi_snapshot_preview1" "args_sizes_get" (func $args (param i32 i32) (result i32)))
        (memory 1)
        (func (export "run") (result i32 i32)
            call $nested
            i32.const 0 i32.const 4 call $args drop
            i32.const 0 i32.load))"#,
    )
    .unwrap();
    for &outer_tier in Tier::ALL {
        for &inner_tier in Tier::ALL {
            let inner_engine = Engine::new(Config::new().tier(inner_tier)).unwrap();
            let inner_imports =
                wasi_imports(WasiContextBuilder::new().args(["inner", "arg"]).build());
            let inner = Rc::new(RefCell::new(
                Instance::new(
                    &inner_engine,
                    &probe_module("wasi_snapshot_preview1"),
                    &inner_imports,
                )
                .unwrap(),
            ));
            let mut outer_imports = wasi_imports(WasiContextBuilder::new().args(["outer"]).build());
            outer_imports.push(Import::func("host", "nested", move |_, _, out| {
                out[0] = inner.borrow_mut().invoke("argc", &[])?[0];
                Ok(())
            }));
            let outer_engine = Engine::new(Config::new().tier(outer_tier)).unwrap();
            let mut outer = Instance::new(&outer_engine, &outer_wasm, &outer_imports).unwrap();
            assert_eq!(
                outer.invoke("run", &[]).unwrap(),
                vec![Value::I32(2), Value::I32(1)]
            );
        }
    }
}

#[test]
fn wasi_import_signatures_are_checked_at_link_time() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let imports = wasi_imports(WasiContextBuilder::new().build());
        for declaration in [
            "(import \"wasi_snapshot_preview1\" \"args_get\" (func))",
            "(import \"wasi_snapshot_preview1\" \"clock_time_get\" (func (param i32 i32 i32) (result i32)))",
            "(import \"wasi_snapshot_preview1\" \"proc_exit\" (func (param i32) (result i32)))",
        ] {
            let wasm = wat::parse_str(format!("(module {declaration})")).unwrap();
            let error = Instance::new(&engine, &wasm, &imports).err().expect("wrong signature must not link");
            assert!(error.is_unlinkable(), "{tier:?}: {error}");
        }
    }
}

#[test]
fn file_access_uses_each_contexts_preopened_directory() {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("sf-wasi-context-{}-{suffix}", std::process::id()));
    let first_dir = root.join("first");
    let second_dir = root.join("second");
    std::fs::create_dir_all(&first_dir).unwrap();
    std::fs::create_dir_all(&second_dir).unwrap();
    std::fs::write(first_dir.join("token"), b"A").unwrap();
    std::fs::write(second_dir.join("token"), b"B").unwrap();
    let wasm = wat::parse_str(r#"(module
        (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
        (memory 1)
        (data (i32.const 128) "token")
        (func (export "read") (result i32)
            i32.const 3 i32.const 0 i32.const 128 i32.const 5 i32.const 0
            i64.const 2 i64.const 0 i32.const 0 i32.const 16 call $open
            if unreachable end
            i32.const 0 i32.const 32 i32.store
            i32.const 4 i32.const 1 i32.store
            i32.const 16 i32.load i32.const 0 i32.const 1 i32.const 8 call $read
            if unreachable end
            i32.const 16 i32.load call $close if unreachable end
            i32.const 32 i32.load8_u))"#).unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut first = Instance::new(
            &engine,
            &wasm,
            &wasi_imports(
                WasiContextBuilder::new()
                    .preopen_dir(".", &first_dir)
                    .build(),
            ),
        )
        .unwrap();
        let mut second = Instance::new(
            &engine,
            &wasm,
            &wasi_imports(
                WasiContextBuilder::new()
                    .preopen_dir(".", &second_dir)
                    .build(),
            ),
        )
        .unwrap();
        for _ in 0..2 {
            assert_eq!(result(&mut first, "read"), b'A' as i32);
            assert_eq!(result(&mut second, "read"), b'B' as i32);
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
