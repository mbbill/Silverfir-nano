//! Declared and host-provided resource limits preserve growth results.
use sf_nano_core::{Config, Engine, Import, Instance, Limits, Tier, Value};

#[test]
fn memory_growth_respects_limits_and_rejects_unrepresentable_requests() {
    for &tier in Tier::ALL {
        for width in [32, 64] {
            for imported in [false, true] {
                for maximum in [None, Some(2)] {
                    let ty = format!("i{width}");
                    let width_marker = if width == 64 { "i64 " } else { "" };
                    let maximum_text = maximum.map_or(String::new(), |n| format!(" {n}"));
                    let memory = format!("(memory {width_marker}1{maximum_text})");
                    let declaration = if imported {
                        format!("(import \"host\" \"memory\" {memory})")
                    } else {
                        memory
                    };
                    let wasm = wat::parse_str(format!(
                        "(module {declaration}
                            (func (export \"grow\") (param {ty}) (result {ty})
                                local.get 0 memory.grow)
                            (func (export \"size\") (result {ty}) memory.size))"
                    ))
                    .unwrap();
                    let limits = if width == 64 {
                        Limits::new_64(1, maximum)
                    } else {
                        Limits::new(1, maximum)
                    }
                    .unwrap();
                    let imports = if imported {
                        vec![Import::memory_with_limits("host", "memory", limits)]
                    } else {
                        vec![]
                    };
                    let engine =
                        Engine::new(Config::new().tier(tier).wasm_memory_max_pages(u32::MAX))
                            .unwrap();
                    let mut instance = Instance::new(&engine, &wasm, &imports).unwrap();
                    let value = |n: i64| {
                        if width == 64 {
                            Value::I64(n)
                        } else {
                            Value::I32(n as i32)
                        }
                    };
                    assert_eq!(
                        instance.invoke("grow", &[value(1)]).unwrap(),
                        vec![value(1)]
                    );
                    assert_eq!(
                        instance.invoke("grow", &[value(0)]).unwrap(),
                        vec![value(2)]
                    );
                    if maximum.is_some() {
                        assert_eq!(
                            instance.invoke("grow", &[value(1)]).unwrap(),
                            vec![value(-1)],
                            "tier={tier:?}, width={width}, imported={imported}, max={maximum:?}, delta=1"
                        );
                    }
                    // These fail before allocation: memory32 exceeds its Wasm
                    // page cap; memory64 cannot represent the byte count.
                    let huge = if width == 64 { 1_i64 << 48 } else { 65536 };
                    for delta in [huge, -1] {
                        assert_eq!(
                            instance.invoke("grow", &[value(delta)]).unwrap(),
                            vec![value(-1)],
                            "tier={tier:?}, width={width}, imported={imported}, max={maximum:?}, delta={delta}"
                        );
                        assert_eq!(instance.invoke("size", &[]).unwrap(), vec![value(2)]);
                    }
                }
            }
        }
    }
}

#[test]
fn table_growth_preserves_explicit_and_unspecified_host_limits() {
    for &tier in Tier::ALL {
        for width in [32, 64] {
            for imported in [false, true] {
                for maximum in [None, Some(2)] {
                    let maximum_text = maximum.map_or(String::new(), |n| format!(" {n}"));
                    let ty = format!("i{width}");
                    let width_marker = if width == 64 { "i64 " } else { "" };
                    let table = format!("(table {width_marker}1{maximum_text} funcref)");
                    let declaration = if imported {
                        format!("(import \"host\" \"table\" {table})")
                    } else {
                        table
                    };
                    let wasm = wat::parse_str(format!(
                        "(module {declaration}
                        (func (export \"grow\") (param {ty}) (result {ty})
                            ref.null func local.get 0 table.grow)
                        (func (export \"size\") (result {ty}) table.size))"
                    ))
                    .unwrap();
                    let imports = if imported {
                        vec![Import::table_with_limits(
                            "host",
                            "table",
                            if width == 64 {
                                Limits::new_64(1, maximum).unwrap()
                            } else {
                                Limits::new(1, maximum).unwrap()
                            },
                        )]
                    } else {
                        vec![]
                    };
                    let value = |n: i64| {
                        if width == 64 {
                            Value::I64(n)
                        } else {
                            Value::I32(n as i32)
                        }
                    };
                    let engine = Engine::new(Config::new().tier(tier)).unwrap();
                    let mut instance = Instance::new(&engine, &wasm, &imports).unwrap();
                    assert_eq!(
                        instance.invoke("grow", &[value(1)]).unwrap(),
                        vec![value(1)]
                    );
                    assert_eq!(
                        instance.invoke("grow", &[value(0)]).unwrap(),
                        vec![value(2)]
                    );
                    if maximum.is_some() {
                        assert_eq!(
                            instance.invoke("grow", &[value(1)]).unwrap(),
                            vec![value(-1)]
                        );
                    }
                    // These exceed the index limit or Vec's representable
                    // capacity; they do not require a large allocation.
                    let huge = if width == 64 { 1_i64 << 60 } else { -1 };
                    for delta in [huge, -1] {
                        assert_eq!(
                            instance.invoke("grow", &[value(delta)]).unwrap(),
                            vec![value(-1)],
                            "tier={tier:?}, width={width}, imported={imported}, max={maximum:?}, delta={delta}"
                        );
                    }
                    assert_eq!(instance.invoke("size", &[]).unwrap(), vec![value(2)]);
                }
            }
        }
    }
}
