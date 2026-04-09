# RP2350 / Pico 2 Bring-Up Plan

Status: draft plan.
Audience: anyone adding a new native backend and a bare-metal platform
integration for RP2350 / Raspberry Pi Pico 2.

## 1. Goal

Bring up Silverfir-nano as a **real native Wasm JIT** on the Arm side of
RP2350 / Pico 2, with a path that is credible on a small MCU and does not
assume Linux, an MMU, guard pages, or abundant RAM.

The first public milestone is:

1. Boot on real Pico 2 hardware.
2. Allocate a small executable code arena in SRAM.
3. JIT a tiny Wasm module to native Thumb code.
4. Run it and report the result over UART/USB serial or a simple LED/UART
   harness.

This plan is intentionally **Arm-core first**. RP2350 also has the Hazard3
RISC-V cluster, but that should be treated as a separate follow-up target.

## 2. Target Definition

### Chosen Rust target

Start with:

- `thumbv8m.main-none-eabi`
- `-C target-cpu=cortex-m33`

### Why this exact target

- It matches the Arm core we actually want to use on RP2350.
- It keeps the ABI simple at first: soft-float ABI, no immediate obligation to
  model FP register calling conventions in generated code.
- It still leaves the door open to use M33 instructions and DSP-friendly
  integer codegen where useful.

### Explicit non-goals for v1

- No Hazard3 / RISC-V support.
- No FPU-register codegen.
- No hard-float ABI (`thumbv8m.main-none-eabihf`) in the first milestone.
- No guard-page-backed linear memory.
- No broad Cortex-M family abstraction on day one if it slows the RP2350 path.

## 3. Current Repo Constraints

These are the first things to fix or route around.

### 3.1 32-bit Arm is currently conflated with `armv7a`

`sf-nano-core/build.rs` currently maps every `target_arch = "arm"` build to
`sf_arch_armv7a`. That is correct for the existing ARMv7-A backend, but wrong
for Cortex-M / Thumb-only targets.

Implication:

- a `thumbv8m.main-none-eabi` build would currently select the wrong backend
  family
- the first bring-up step is not codegen, it is **backend-selection plumbing**

### 3.2 Bare-metal executable memory exists as a contract, not a platform

`sf-nano-core/src/vm/runtime/os/none.rs` already defines the right embedder
contract:

- `sf_os_alloc_executable`
- `sf_os_free_executable`
- `sf_os_begin_write_executable`
- `sf_os_finish_write_executable`

This is good news: RP2350 does not need a new runtime model, only a concrete
implementation of the existing bare-metal one.

### 3.3 The default code buffer size is MCU-hostile

`sf-nano-core/src/vm/runtime/code_buf.rs` defaults to:

- `12 * 1024 * 1024` on 32-bit targets
- `16 * 1024 * 1024` on 64-bit targets

That is fine for hosted systems and wrong for Pico 2. RP2350 / Pico 2
bring-up must introduce a **small, explicit code-cache budget** instead of
relying on the current default.

## 4. Design Decisions For This Bring-Up

### 4.1 Backend family name

Do not reuse `armv7a`.

Recommended new backend/config vocabulary:

- `sf_arch_thumbm` or `sf_arch_cortex_m`
- module path: `sf-nano-core/src/vm/arch/thumbm/`

Reason:

- the important split is not "32-bit Arm" but "`armv7a` A-profile" versus
  "Thumb-only M-profile"
- this keeps future M0+/M4/M7 work possible without pretending the ISA is the
  same as ARMv7-A

### 4.2 Start with integer-first codegen

The first backend should support:

- `i32` arithmetic and compares
- branches
- loads/stores
- local calls / returns
- enough `i64` support to run small real Wasm, using helper calls where needed

The first backend should not try to solve everything at once:

- keep FP register allocation out of v1
- allow helper calls for difficult `i64` and float operations
- inline common integer fast paths first, optimize later

### 4.3 Prefer a tiny real demo over a broad feature surface

The first success case should be a small Wasm payload that demonstrates:

- native code emission
- executable memory on SRAM
- correct return values
- some control flow and memory traffic

That is enough to prove "tiny-device JIT" before chasing full benchmark parity.

## 5. Work Breakdown

### Phase 0: Scope Lock

Deliverable:

- a written backend scope and success criteria, captured in this doc

Decisions to lock:

- Arm cluster only
- `thumbv8m.main-none-eabi`
- `cortex-m33`
- soft-float ABI
- integer-first backend
- SRAM code arena with explicit small capacity

### Phase 1: Backend Selection Plumbing

Goal:

- allow Cortex-M builds to select a new backend cleanly

Tasks:

1. Split the current `target_arch = "arm"` handling in `build.rs`.
2. Add a new cfg for the Cortex-M / Thumb-M backend.
3. Update `sf-nano-core/src/vm/arch/mod.rs` to recognize the new backend.
4. Make sure existing `armv7a` builds stay unchanged.

Success criteria:

- `cargo check --target thumbv8m.main-none-eabi` selects the new backend path
- existing `armv7a` target selection is unaffected

### Phase 2: Backend Skeleton

Goal:

- create a compileable backend that can emit at least a minimal function body

Tasks:

1. Add `vm/arch/thumbm/` with the same coarse structure as the other
   backends:
   - `abi.rs`
   - `backend.rs`
   - `control.rs`
   - `enc.rs`
   - `inst.rs`
   - `mod.rs`
   - `reg.rs`
   - `compile.rs`
2. Define the register set and calling convention assumptions for M33.
3. Decide the preserved vs scratch register model for the backend.
4. Define how helper calls are made from generated Thumb code.

Important issue:

- function pointers in Thumb state carry the low-bit state tag; any entry-point
  pointer or helper-call target used as a code pointer must respect that
  convention

Success criteria:

- backend compiles
- backend can emit a trivial leaf function and return to the caller

### Phase 3: Minimal Instruction Set

Goal:

- run tiny Wasm with no platform dependencies beyond executable SRAM

Instruction priorities:

1. move / materialize constants
2. `add`, `sub`, `and`, `or`, `xor`
3. shifts
4. integer compare + branch
5. loads / stores for linear memory and locals
6. direct call / return
7. trap tail / error propagation

Expected fallback policy:

- use helper calls for hard `i64` ops initially
- use helper calls for float ops initially
- only inline more operations after the end-to-end path is working

Success criteria:

- small integer-only Wasm samples JIT and run correctly

### Phase 4: RP2350 Bare-Metal Runtime Shim

Goal:

- provide the concrete `sf_os_none` implementation for Pico 2

Tasks:

1. Create a tiny RP2350/Pico 2 embedder crate or board harness outside the
   workspace fast path.
2. Implement:
   - `sf_os_alloc_executable`
   - `sf_os_free_executable`
   - `sf_os_begin_write_executable`
   - `sf_os_finish_write_executable`
3. Back these with a fixed SRAM arena reserved for JIT code.
4. Decide whether `finish_write` needs only barriers (`dsb`/`isb`) or extra
   cache/prefetch maintenance on RP2350; validate this on hardware.

Policy for v1:

- no W^X enforcement required if the hardware/runtime model does not support it
- simple static arena is acceptable
- the arena size should be explicit, small, and board-tuned

Success criteria:

- JIT code buffer allocation works on real hardware
- emitted code becomes executable and observable by the fetch path

### Phase 5: Code-Cache Budgeting

Goal:

- make the engine realistic on a 520 KB SRAM-class MCU

Tasks:

1. Replace or bypass the 12 MB 32-bit default code buffer.
2. Add a platform-configurable code cache size for MCU targets.
3. Start with a conservative code cache, for example:
   - `32 KB` minimum viable
   - `64-128 KB` comfortable demo range
4. Add clear failure behavior when code cache exhaustion occurs.

Success criteria:

- the engine does not accidentally assume multi-megabyte executable memory

### Phase 6: Emulator-Assisted Backend Bring-Up

Goal:

- debug ISA/backend correctness before depending on RP2350 hardware

Use:

- `qemu-system-arm -machine mps2-an505 -cpu cortex-m33`

Use QEMU for:

- backend instruction encoding validation
- helper-call ABI validation
- function entry/exit correctness
- branch, trap, and register-save debugging

Do not use QEMU as a substitute for Pico 2 for:

- RP2350 boot assumptions
- memory map assumptions
- flash/XIP assumptions
- peripheral integration

Success criteria:

- backend can pass a small native-function smoke path under generic M33

### Phase 7: Real Pico 2 Bring-Up

Goal:

- move from generic M33 correctness to actual RP2350 execution

Recommended order:

1. Boot a trivial firmware that prints or blinks.
2. Link Silverfir with the bare-metal RP2350 harness.
3. Bring up executable SRAM allocation.
4. Run a hard-coded tiny Wasm module.
5. Add a few more Wasm smoke cases:
   - integer arithmetic
   - branching
   - local calls
   - linear-memory load/store
6. Record code size, code-cache usage, and total SRAM footprint.

Success criteria:

- real hardware demo of JIT-compiled Wasm

### Phase 8: Stabilization

After the first successful hardware demo:

1. inline more `i64` operations where it materially helps
2. add better constant/literal-pool handling
3. reduce helper-call frequency
4. tune register allocation for Thumb/M-profile constraints
5. decide whether FP support is worth adding to this backend

## 6. Validation Matrix

Minimum test ladder:

1. `cargo check` for the new target and backend
2. backend-only smoke under QEMU M33
3. RP2350 firmware boots with runtime shim linked
4. one leaf Wasm function returns correct result
5. one branching Wasm function returns correct result
6. one Wasm memory test passes
7. one local-call Wasm test passes

Recommended first demo module:

- tiny integer workload
- no imports
- no floats
- no large memory

Examples:

- add / mul / compare
- small loop
- tiny recursive or local-call tree if stack behavior needs proving

## 7. Risks

### High risk

- Thumb entry-point / function-pointer state-bit bugs
- code emission correctness under Thumb-only control-flow rules
- underestimating the work to split `armv7a` assumptions out of the current
  32-bit Arm path
- executable-memory visibility bugs on real hardware

### Medium risk

- literal-pool placement and range handling
- helper-call ABI mismatches
- code-cache fragmentation if the first allocator is too naive
- unexpectedly large Wasm linear-memory overhead relative to the code size

### Low risk

- the bare-metal runtime interface itself; the repo already has the right
  abstraction for it

## 8. Open Questions

These should be answered early, but they should not block Phase 1.

1. Should the backend be named narrowly (`cortex_m33`) or as a family
   (`thumbm`)?
2. Does RP2350 require any instruction-fetch maintenance beyond barriers after
   self-modifying code writes into SRAM?
3. Should v1 expose the code-cache size as a constructor/runtime option or keep
   it platform-fixed?
4. Is the first public demo better as:
   - a board firmware with built-in Wasm bytes, or
   - a host-fed Wasm payload over USB/UART?

## 9. Recommended First Milestone

Do not define success as "full backend."

Define the first success as:

1. new Cortex-M backend selected by target cfg
2. minimal Thumb code emission works
3. RP2350 harness provides executable SRAM
4. one tiny Wasm module JITs and runs on Pico 2

Once that exists, the project can honestly show:

"This is not just `no_std`; it is a real Wasm JIT running on a microcontroller-class device."
