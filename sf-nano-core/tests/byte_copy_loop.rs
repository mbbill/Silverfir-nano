//! Byte-copy idiom recovery must preserve overlap, empty ranges and trap writes.
use sf_nano_core::{Config, Engine, Instance, Value};

const COPY_LOOP: &str = r#"(module
    (memory (export "memory") 1)
    (func (export "copy") (param $dst i32) (param $src i32) (param $end i32)
        (local $index i32) (local $step i32)
        (local.set $index (i32.shl (memory.size) (i32.const 16)))
        (if (i32.or
                (i32.lt_u (local.get $index) (i32.add (local.get $dst) (local.get $end)))
                (i32.gt_u (i32.add (local.get $src) (local.get $end)) (local.get $index)))
            (then unreachable))
        (local.set $step (if (result i32) (i32.gt_u (local.get $dst) (local.get $src))
            (then
                (local.set $index (i32.sub (local.get $end) (i32.const 1)))
                (local.set $end (i32.const -1))
                (i32.const -1))
            (else
                (local.set $index (i32.const 0))
                (i32.const 1))))
        (block $done
            (loop $copy
                (br_if $done (i32.eq (local.get $index) (local.get $end)))
                (i32.store8
                    (i32.add (local.get $dst) (local.get $index))
                    (i32.load8_u (i32.add (local.get $src) (local.get $index))))
                (local.set $index (i32.add (local.get $index) (local.get $step)))
                (br $copy)))))"#;

fn reference(memory: &mut [u8], dst: u32, src: u32, len: u32) -> bool {
    // The guest helper checks endpoints with Wasm i32 arithmetic. Wrapped
    // endpoints can pass this check and still trap part way through the loop.
    if dst.wrapping_add(len) > memory.len() as u32 || src.wrapping_add(len) > memory.len() as u32 {
        return false;
    }
    let (mut index, end, step) = if dst > src {
        (len.wrapping_sub(1), u32::MAX, u32::MAX)
    } else {
        (0, len, 1)
    };
    while index != end {
        let Some(&byte) = memory.get(src.wrapping_add(index) as usize) else {
            return false;
        };
        let Some(slot) = memory.get_mut(dst.wrapping_add(index) as usize) else {
            return false;
        };
        *slot = byte;
        index = index.wrapping_add(step);
    }
    true
}

#[test]
fn copying_matches_byte_loop_including_partial_writes_before_trap() {
    let wasm = wat::parse_str(COPY_LOOP).unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let initial: Vec<u8> = (0..65536).map(|n| ((n * 37) ^ (n >> 8)) as u8).collect();
    let mut cases = vec![
        (u32::MAX, u32::MAX, 0),
        (0, 65536, 0),
        (0, 65535, 2),
        (65535, 0, 2),
        (65534, 0, 2),
        (0xfffffff0, 32, 32),
        (32, 0xfffffff0, 32),
        (0xfffffffe, 0xfffffffd, 4),
    ];
    for dst in [0, 1, 15, 32, 63, 128] {
        for src in [0, 1, 16, 32, 64, 128] {
            for len in [0, 1, 7, 32, 65, 256] {
                cases.push((dst, src, len));
            }
        }
    }
    // Use one compiled module for ordinary calls; a trap must leave it usable
    // for the following call as well as preserving the exact memory contents.
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    for (dst, src, len) in cases {
        instance.memory_mut().unwrap().copy_from_slice(&initial);
        let mut expected = initial.clone();
        let success = reference(&mut expected, dst, src, len);
        let result = instance.invoke(
            "copy",
            &[
                Value::I32(dst as i32),
                Value::I32(src as i32),
                Value::I32(len as i32),
            ],
        );
        assert_eq!(
            result.is_ok(),
            success,
            "copy({dst}, {src}, {len}): {result:?}"
        );
        assert_eq!(
            instance.memory().unwrap(),
            expected.as_slice(),
            "copy({dst}, {src}, {len})"
        );
    }
}
