//! Narrow address reads preserve sign extension, trap order and instance reuse.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn narrow_read_loops_match_memory_and_preserve_the_last_write_before_trapping() {
    let mut wat = String::from("(module (memory (export \"memory\") 1)");
    for (name, op) in [
        ("u8", "i64.load8_u"),
        ("s8", "i64.load8_s"),
        ("u16", "i64.load16_u"),
        ("s16", "i64.load16_s"),
    ] {
        wat.push_str(&format!(
            r#"(func (export "{name}") (param $p i32) (param $n i32) (result i64)
                (local $sum i64)
                (block $done (loop $next
                    (br_if $done (i32.eqz (local.get $n)))
                    (local.set $sum (i64.add (local.get $sum) ({op} offset=1 (local.get $p))))
                    (i64.store (i32.const 0) (local.get $sum))
                    (local.set $p (i32.add (local.get $p) (i32.const 1)))
                    (local.set $n (i32.sub (local.get $n) (i32.const 1)))
                    (br $next)))
                (local.get $sum))"#
        ));
    }
    wat.push(')');
    let wasm = wat::parse_str(&wat).unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    let initial: Vec<u8> = (0..65536).map(|n| ((n * 71) ^ (n >> 8)) as u8).collect();
    for (name, width, signed) in [
        ("u8", 1, false),
        ("s8", 1, true),
        ("u16", 2, false),
        ("s16", 2, true),
    ] {
        for (start, count) in [
            (32u32, 127u32),
            (65532, 5),
            (0xffff_ffff, 1),
            (0x8000_0000, 1),
            (65534, 1),
            (32, 1),
            (65535, 0),
        ] {
            instance.memory_mut().unwrap().copy_from_slice(&initial);
            let mut expected = initial.clone();
            let mut sum = 0i64;
            let mut success = true;
            for index in 0..count {
                let address = u64::from(start.wrapping_add(index)) + 1;
                let bytes = usize::try_from(address).ok().and_then(|address| {
                    address
                        .checked_add(width)
                        .and_then(|end| expected.get(address..end))
                });
                let Some(bytes) = bytes else {
                    success = false;
                    break;
                };
                let value = match (width, signed) {
                    (1, false) => i64::from(bytes[0]),
                    (1, true) => i64::from(bytes[0] as i8),
                    (2, false) => i64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
                    (2, true) => i64::from(i16::from_le_bytes([bytes[0], bytes[1]])),
                    _ => unreachable!(),
                };
                sum = sum.wrapping_add(value);
                expected[..8].copy_from_slice(&sum.to_le_bytes());
            }
            let actual =
                instance.invoke(name, &[Value::I32(start as i32), Value::I32(count as i32)]);
            assert_eq!(
                actual.is_ok(),
                success,
                "{name}({start}, {count}): {actual:?}"
            );
            if success {
                assert_eq!(
                    actual.unwrap(),
                    vec![Value::I64(sum)],
                    "{name}({start}, {count})"
                );
            }
            assert_eq!(
                instance.memory().unwrap(),
                expected.as_slice(),
                "{name}({start}, {count})"
            );
        }
    }
}
