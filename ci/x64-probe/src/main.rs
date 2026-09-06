use sf_nano_core::{Caller, Config, Engine, Import, Instance, Tier, Value};
use std::time::{Duration, Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let wasm = wat::parse_file(&args[1]).unwrap();
    let mode = &args[2];
    let input: i64 = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(0);
    let seconds: f64 = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(0.0);
    let clock = Instant::now();
    let imports = [Import::func(
        "env",
        "clock_ms",
        move |_: &mut Caller, _: &[Value], out: &mut [Value]| {
            out[0] = Value::I32(clock.elapsed().as_millis() as i32);
            Ok(())
        },
    )];
    let engine = Engine::new(Config::new().tier(Tier::Jit).parallel_compilation(false)).unwrap();
    let mut instance = Instance::new(
        &engine,
        &wasm,
        if mode == "coremark" { &imports } else { &[] },
    )
    .unwrap();
    let params = match mode.as_str() {
        "coremark" => vec![],
        "i64" => vec![Value::I64(input)],
        "i32" => vec![Value::I32(input as i32)],
        "setup" => instance
            .invoke("setup", &[Value::I32(input as i32)])
            .unwrap()
            .into_iter()
            .collect(),
        _ => panic!("unknown mode {mode}"),
    };
    let start = Instant::now();
    let mut count = 0u64;
    loop {
        let result = instance.invoke("run", &params).unwrap();
        std::hint::black_box(&result);
        count += 1;
        if start.elapsed() >= Duration::from_secs_f64(seconds) {
            eprintln!(
                "runs={count} elapsed={:?} result={result:?}",
                start.elapsed()
            );
            break;
        }
    }
    if mode == "setup" {
        if args[1].contains("argon2") {
            let result = instance.invoke("output", &params).unwrap();
            assert_eq!(result[0], Value::I64(0x4CDBBC7DE0EAA94));
        }
        instance.invoke("teardown", &params).unwrap();
    }
}
