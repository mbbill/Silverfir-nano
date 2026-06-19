- Guard-page faults raised by out-of-bounds memory accesses are caught by a
  process-level signal handler that maps the faulting native address back to a
  Wasm trap and resumes the runtime's trap path (`trap_signal`), available only
  on configurations that have guard pages.

- The engine keeps a registry of JIT code ranges that a fault handler consults
  to decide whether a faulting address belongs to generated code, and exposes a
  reset hook for harnesses that repeatedly build and drop native modules.

- The OS-agnostic trap table, storm counter, and install latch are separated
  from the per-(arch × os) ucontext/register surgery, which lives behind a single
  `install_platform_handler` entry point with one signal submodule per platform.

## Facts

- 2026-03-15 (3b6d2f59) rationale: the handler must be async-signal-safe so it
  allocates nothing — it reads the faulting PC and the X19 context pointer
  straight from the platform ucontext thread-state, sets the trap_kind flag at a
  registration-time-supplied offset and redirects PC to return_error rather than
  constructing the WasmError in-signal, leaving the Rust caller to build the error
  after the native frame unwinds; a faulting PC outside every registered JIT range
  aborts the process (code).

- 2026-03-16 (3b1facfd) pitfall: the JIT code-range registry and the signal
  handler's storm counter are process-global and do not track module lifetimes,
  so a harness that constructs and drops many native modules must clear the
  registered ranges and reset the signal counter between independent runs
  (reset_native_runtime_state) — otherwise a later trap could resolve a faulting
  PC against a stale dropped range, and accumulated signal counts across many
  expected-trap cases would trip the >100-consecutive-signals abort guard (code).

- 2026-03-16 (3b1facfd) pitfall: the Darwin arm64 mcontext64 __es (exception
  state) block is 16 bytes (far:u64 + esr:u32 + exception:u32), not 24; the
  hardcoded MCONTEXT_SS_OFFSET was 24 and read the thread state at the wrong
  offset, recovering a wrong faulting PC/X19 — corrected to 16 to match the
  x86_64 path which already used 16 (code).

- 2026-04-11 (524024e8) pitfall: the Linux x86_64 guard-page handler installed
  its sigaction through a hand-rolled `kernel_sigaction` whose field order and
  types did not match the glibc/musl userspace `struct sigaction`, so the raw
  sigaction call read the wrong bytes for mask/flags/restorer; the fix mirrors the
  exact userspace layout (sa_mask a 128-byte sigset_t directly after
  sa_sigaction, sa_flags an i32 at offset 136, sa_restorer at 144, size 152),
  pinned by a compile-time offset/size assertion — a bare extern sigaction binding
  must mirror the userspace layout, not the kernel's rt_sigaction ABI (code).

- 2026-04-15 (b3b54d62) rationale: the Windows x86_64 trap handler is a Vectored
  Exception Handler registered first in the chain (AddVectoredExceptionHandler(1,
  ...)) so it sees access violations before any user handler; unlike the POSIX
  handlers it must return EXCEPTION_CONTINUE_SEARCH for any fault whose RIP does
  not resolve to a JIT region rather than aborting, because a VEH observes every
  exception in the whole process and aborting on non-JIT faults would break
  legitimate host exception flows (debuggers, the OS unhandled-exception filter);
  only JIT-attributed faults are counted against the storm guard and resumed by
  patching RAX=1 and RIP=return_error_label (code).

## Moves

- 2026-04-07 (11b835a2) replaced [[monolithic-trap-signal]]: the guard-page
  signal handler held every platform's ucontext layout and register surgery
  inline in one file, so the OS-agnostic trap table was entangled with
  per-(arch x os) frame parsing; splitting the platform half into os/signal/
  modules behind a single install_platform_handler entry point lets trap_signal
  own only the trap table, storm counter, and install latch, and a new platform is
  one more signal submodule (code).
