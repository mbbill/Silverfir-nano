//! Counts-probe runner: instantiate a module on the JIT, call
//! `setup(400)` once, then `run(data)` a requested number of times.
//! Built against two engine revisions and run under cachegrind, the
//! instruction/miss/branch counts separate added work from front-end
//! and layout effects that wall time alone cannot.

use sf_nano_core::{Config, Engine, Instance, Tier, Value};

fn main() {
    let mut args = std::env::args().skip(1);
    let wasm_path = args.next().expect("usage: counts-runner <wasm> <reps>");
    let reps: u32 = args
        .next()
        .expect("usage: counts-runner <wasm> <reps>")
        .parse()
        .expect("reps must be a non-negative integer");
    let wasm = std::fs::read(&wasm_path).expect("failed to read wasm file");

    let config = Config::new().tier(Tier::Jit).parallel_compilation(false);
    let engine = Engine::new(config).expect("failed to configure engine");
    let mut instance = Instance::new(&engine, &wasm, &[]).expect("failed to instantiate");

    let results = instance
        .invoke("setup", &[Value::I32(400)])
        .expect("setup failed");
    let Some(Value::I32(data)) = results.first().copied() else {
        panic!("setup returned no i32");
    };
    for _ in 0..reps {
        instance
            .invoke("run", &[Value::I32(data)])
            .expect("run failed");
    }
}
