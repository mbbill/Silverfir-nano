//! The WAST driver and its support modules are gated on `jit`: the shared
//! runner drives entity-model operations the JIT provides, and the
//! interpreter is exercised through the same runner in a `jit,interp`
//! build. A pure-`interp` build is compile coverage for that feature
//! boundary; it states so instead of shipping a silent half-binary.
#[cfg(feature = "jit")]
mod discovery;
#[cfg(feature = "jit")]
mod driver;
#[cfg(feature = "jit")]
mod summary;
#[cfg(feature = "jit")]
mod types;
// One runner for both engines. It is gated on `jit` because it drives
// entity-model operations the JIT provides; the interpreter answers what
// it can and reports the rest, which is how the gap between them stays
// measurable instead of becoming two suites that disagree.
#[cfg(feature = "jit")]
mod wast_test_runner;

#[cfg(feature = "jit")]
fn main() {
    driver::run()
}

#[cfg(not(feature = "jit"))]
fn main() {
    eprintln!(
        "this sf-nano-spectest build has no WAST driver: the shared runner is \
         gated on the `jit` feature. Rebuild with `--features jit` or \
         `--features jit,interp` to run the suite."
    );
    std::process::exit(2);
}
