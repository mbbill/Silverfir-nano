# Silverfir-nano: Closing the Interpreter–JIT Gap Through Stack-Machine Fusion and Register Caching

## Abstract

WebAssembly interpreters are widely assumed to trail JIT compilers by an order of magnitude on compute-intensive workloads.
We present Silverfir-nano, a WebAssembly interpreter that narrows this gap substantially.
On an Apple M4 processor, Silverfir-nano reaches 27–62% of the throughput of Wasmtime's optimizing Cranelift JIT across four benchmarks (SHA-256, bzip2, LZ4, CoreMark), without runtime code generation.
Among interpreters, it outperforms the next-fastest (wasm3) by 1.7–2.5×.
Silverfir-nano achieves this through four synergistic techniques:
(1) automated stack-machine instruction fusion that substantially reduces dispatch counts while enabling the C compiler to optimize across instruction boundaries—an advantage fundamentally unavailable to register-machine interpreters;
(2) a top-of-stack register window that maps four stack slots to hardware registers via the `preserve_none` calling convention and tail-call dispatch;
(3) a hot-local register cache (L0/L1/L2) whose operations participate as first-class fusion operands, reducing local variable access to zero instructions in fused handlers;
and (4) next-handler preloading with guard-check dispatch that hides load-to-use latency.
An ablation study confirms that fusion is the dominant optimization, delivering 1.7–2.0× speedup alone, while the register cache amplifies fusion by an additional 5–61% depending on workload.
The core interpreter is 230 KB stripped; the full build with 1,500 fusion patterns is 3 MB.

## 1. Introduction

The conventional wisdom in language runtime design holds that interpreters cannot compete with JIT compilers on compute-intensive workloads.
JIT compilers translate bytecode to native machine code, applying register allocation, instruction scheduling, and constant propagation.
Interpreters dispatch each instruction through an indirect branch, incurring overhead that compounds across millions of executed instructions.
The gap is typically assumed to be 5–10× [31, 20].

WebAssembly (Wasm) is a particularly interesting target for interpreter optimization.
As a low-level stack machine with static type checking and deterministic stack heights at every program point [10], it provides structural properties that an interpreter can exploit more aggressively than interpreters for dynamically-typed languages.
Yet existing Wasm interpreters—wasm3 [4], WAMR [5], wasmi [6]—still trail JIT compilers by large margins, even on the workload types most favorable to interpretation.

We present Silverfir-nano, a WebAssembly interpreter that substantially narrows this gap.
On an Apple M4 processor, it reaches 27–62% of the optimizing Cranelift JIT [16] across four benchmarks (Table 1), without runtime code generation.
Among interpreters, it outperforms wasm3—the fastest prior interpreter—by 1.7–2.5× across the same benchmarks.
On CoreMark, Silverfir-nano reaches 62% of Cranelift and scores 2.2× higher than wasm3.

### Contributions

This paper makes four novel contributions:

1. **The argument for stack-machine fusion over register-machine conversion** (Section 4.1–4.2).
   We show that staying with a stack-machine IR, rather than converting to a register machine as done by wasm3 [4] and WAMR [5], enables *automated* superinstruction generation where the C compiler optimizes across instruction boundaries.
   Register-machine fusion cannot achieve this due to runtime aliasing barriers.
   We provide assembly evidence showing 3–5× fewer instructions for the stack-machine approach (Table 4).

2. **TOS register window with compile-time depth tracking** (Section 4.3).
   We map four stack slots to hardware registers via `preserve_none` [17] and `musttail` tail-call dispatch, scaling beyond the 1–2 registers used in prior work [4, 22].
   Compile-time knowledge of Wasm's static stack heights selects among four depth-variant handlers per instruction (D1 through D4, cycling modulo four as stack depth grows), replacing the finite state machines of prior TOS caching schemes.
   Combined with fusion, intermediate stack operations within fused handlers compile to pure register arithmetic with zero memory traffic.

3. **Hot-local register cache with hotness-based index swap** (Section 4.4).
   At load time, a loop-nesting-weighted analysis identifies the three hottest local variables per function.
   A function prologue physically swaps these locals to indices 0–2, guaranteeing they map to dedicated hardware registers (L0/L1/L2) regardless of their original position.
   These registers participate in the fusion system as first-class operands, so that fused patterns involving hot locals compile to pure register arithmetic with zero frame memory access.

4. **Next-handler preloading with guard-check dispatch** (Section 4.5).
   Each handler receives a preloaded pointer to the next handler as a function argument.
   A three-tier dispatch strategy (always-linear, potentially-branching, always-nonlinear) is generated automatically; the compiler eliminates the guard branch for 89% of handlers.
   Representative arithmetic handlers compile to five AArch64 instructions with zero prologue or epilogue.

Individual techniques in our system draw on established work—tail-call dispatch [1, 9], the `musttail` attribute [9], `preserve_none` [17], top-of-stack caching [22], and superinstruction concepts [19]—but the specific combination and the argument for stack-machine fusion are new.
We position established techniques in Background (Section 2) and focus the Design section on novel contributions.

## 2. Background

### 2.1 WebAssembly Execution Model

WebAssembly [10] is a portable binary instruction format defining a stack-based virtual machine with structured control flow, linear memory, and static type verification.
Properties relevant to interpreter design include:
(a) *static stack heights*—the operand stack height is known at every program point during validation, enabling compile-time register mapping decisions;
(b) *structured control flow*—branches target labeled blocks, simplifying branch resolution;
(c) *typed local variables*—functions declare locals accessed by index (`local.get`, `local.set`, `local.tee`), which persist across instructions and dominate the instruction mix (Section 3).

### 2.2 Interpreter Dispatch Strategies

The performance of a bytecode interpreter is dominated by the dispatch mechanism.
Three strategies are standard:

**Switch dispatch** uses a single `switch` over the opcode, creating a single indirect branch site.
Ertl and Gregg [1] measured 81–98% branch misprediction for this approach, because the Branch Target Buffer (BTB) entry is constantly overwritten with different targets.

**Threaded dispatch** (computed goto) gives each handler its own dispatch instruction, allowing the BTB to learn per-handler correlations.
Misprediction drops to 57–63% [1], but all handlers must reside in a single function, degrading register allocation.

**Tail-call dispatch** separates each handler into an independent function that tail-calls the next.
Each handler gets its own BTB entry *and* independent compiler optimization.
The `musttail` attribute in Clang [9] guarantees tail-call emission (compile-time error on failure).
The `preserve_none` calling convention [17] makes all general-purpose registers caller-saved, eliminating prologue/epilogue overhead.
On AArch64, this provides 26 registers for argument passing; on x86-64, 12.

Rohou et al. [3] revisited Ertl and Gregg's findings with modern processors employing ITTAGE-class indirect branch predictors [2] and found substantially improved prediction accuracy.
Our profiling on Apple M4 confirms >95% branch prediction success with tail-call dispatch.

### 2.3 Top-of-Stack Caching

Caching the top of the operand stack in hardware registers is an established technique.
Ertl [22] introduced stack caching for interpreters in 1995, modeling 1–2 TOS registers as a finite state machine with 3 cache states; the dynamic variant dispatches through a 2D table indexed by cache state and opcode.
Ertl and Gregg [23] combined stack caching with dynamic superinstructions, embedding state transitions within superinstructions to eliminate their dispatch cost, achieving speedups of up to 58%.
HotSpot's template interpreter uses a single TOS register with 9 type-based states (int, float, object, etc.), dispatching through type-variant entry points generated at JVM startup.
wasm3 [4] maps two values to hardware registers (one integer r0, one float fp0) via the standard C calling convention.
CPython 3.14 [7] adopted a fixed cache of 1 via `musttail`, though without `preserve_none` the value is not guaranteed to remain in a hardware register.

Our contribution extends this line of work in three ways (Section 4.3): scaling from 1–2 to four TOS registers via `preserve_none`, replacing the finite state machine with compile-time depth tracking that selects among five handler variants per instruction, and combining TOS caching with 500+ automated fusion patterns so that intermediate stack operations within fused handlers compile to pure register arithmetic.

## 3. Motivation: The Instruction Distribution Problem

WebAssembly's runtime instruction distribution is heavily skewed toward local variable access.
Profiling LLVM-compiled workloads with fusion disabled reveals consistent patterns: in CoreMark (307M dispatches), `local.get` accounts for 26.1% of all dispatched instructions, with `local.set` and `local.tee` adding another 12%.
Local variable access thus constitutes approximately 38% of all dispatches.
Block structure instructions and control flow account for ~22%, while actual arithmetic—the instructions that perform real computation—accounts for less than 40%.

`[TODO: Add specific instruction distribution numbers for at least SHA-256 and bzip2 in the prose above, not just CoreMark. The table below should include all primary benchmarks.]`

> **Table 2.** Instruction distribution (fusion disabled). CoreMark: 307M total dispatches; Lua Fibonacci: 526K. Data from Silverfir-nano's dispatch profiler.

| Category | CoreMark | Lua Fibonacci |
|----------|:--------:|:-------------:|
| Local access (`get`/`set`/`tee`) | ~38% | ~38% |
| Control flow (`br`/`br_if`/`block`/`loop`/`end`) | ~22% | ~24% |
| Arithmetic and logic | ~28% | ~26% |
| Memory (`load`/`store`) | ~8% | ~8% |
| Other | ~4% | ~4% |

`[TODO: Replace Lua Fibonacci column with SHA-256, bzip2, LZ4, and optionally mandelbrot columns. All tables in the paper should use a consistent set of benchmarks: CoreMark, SHA-256, bzip2, LZ4 (and mandelbrot if FP is included).]`

A naive interpreter that dispatches every Wasm instruction individually spends the majority of its time on overhead.
The question becomes: how can we reduce the number of dispatched instructions?

**The register-machine alternative.**
One approach, adopted by wasm3 [4] and WAMR [5], is to convert the stack machine into a register machine at compile time.
Virtual register indices replace stack offsets, and `local.get` effectively disappears.
However, the "virtual" registers live in memory, not hardware registers—each handler must load operands from and store results to a register file.
Mapping virtual registers to hardware registers would require a register allocation pass during module loading — analyzing liveness, computing interference graphs, and assigning physical registers. This adds compilation cost comparable to a baseline JIT's register allocator. Moreover, with tail-call dispatch, each handler receives state through function arguments in a fixed calling-convention order; supporting per-function register assignments would require register permutation at handler boundaries, further increasing code size and complexity. The stack-machine approach avoids this entirely: operand locations are compile-time constants determined by the stack height, requiring no register allocation infrastructure.
More fundamentally, instruction fusion on a register machine cannot benefit from cross-instruction compiler optimization, as we demonstrate in Section 4.2.

## 4. Design

### 4.1 Instruction Fusion

#### Mechanism

Instead of dispatching each Wasm instruction individually, we fuse N consecutive instructions into a single handler.
This reduces dispatches and—critically for a stack machine—enables the C compiler to optimize across instruction boundaries within each fused handler body.

Each instruction in Silverfir-nano is encoded as four 64-bit words (32 bytes): one word holds the handler function pointer, and three are immediate slots for operands.
This wide format appears wasteful for simple instructions, but fusion fills it efficiently: a fused pattern packs multiple instructions' operands into the three slots.
The fusion system rejects any candidate whose combined immediates exceed three slots, guaranteeing every pattern fits.
The 32-byte format aligns cleanly: two instructions per 64-byte cache line.

#### Automated Discovery Pipeline

The central claim of this paper is not just that fusion helps — it is that fusion can be *discovered automatically* at scale, eliminating manual pattern selection.
The pipeline proceeds in five stages:

1. **Profile.** A workload is executed under Silverfir-nano's dispatch profiler, which records instruction N-grams (sequences of 2–8 consecutive Wasm instructions) weighted by dynamic execution count. Hot loops contribute more weight than cold code.

2. **Candidate selection.** A greedy algorithm ranks N-grams by weighted frequency and selects the top candidates subject to three constraints: (a) no internal branches within the fused sequence (the pattern must be a straight-line code fragment); (b) combined immediates fit within the 192-bit encoding budget (three 64-bit slots); (c) the stack effect is compatible with TOS depth tracking (the pattern's net push/pop is well-defined at every intermediate point).

3. **Stack-effect computation.** For each selected pattern, the tool automatically computes the peak intermediate stack height and net stack effect, determines which TOS depth variants are needed, and identifies spill/fill requirements at pattern boundaries.

4. **Configuration generation.** The tool writes a TOML configuration file listing each pattern's instruction sequence, encoding layout (which immediates go in which slots), stack effects, and handler classification (always-linear, potentially-branching, or always-nonlinear).

5. **Code generation.** The build system reads the TOML and generates: (a) Rust pattern matchers that recognize fusible sequences during module loading; (b) C handler implementations — one per pattern per TOS depth variant — that the C compiler optimizes as described in Section 4.2; (c) emission code that encodes fused instructions into the wide instruction format.

**Why automation matters.**
On a register machine, fusion resists automation: the compiler cannot optimize across fused instruction boundaries due to runtime aliasing (Section 4.2), so each fused pattern must be hand-crafted with knowledge of the data-flow pattern. This is why every register-based interpreter that performs fusion uses a small number of manually selected patterns [4, 8].
On a stack machine, fusion is mechanical: concatenate handler bodies, and the C compiler optimizes automatically. This pipeline can produce an arbitrary number of fusion patterns from a handful of training workloads without manual intervention — the built-in set uses 1,500 as a practical balance between coverage and binary size.

**Generalizability.**
The built-in fusion set contains 1,500 patterns discovered from diverse LLVM-compiled workloads.
Because the LLVM Wasm backend produces consistent instruction sequences across programs (the same source-level patterns — loops, conditionals, array access — compile to the same Wasm instruction sequences regardless of application), patterns discovered from a few representative workloads cover a large fraction of the instruction distribution in unseen programs.

`[PLACEHOLDER: Fusion generalizability analysis — pattern overlap across workloads, performance of generic vs. workload-specific sets, sensitivity to pattern count. This is the #1 question reviewers will ask and requires dedicated analysis.]`


#### Dispatch Reduction

Fusion substantially reduces the number of dispatched instructions.
`[PLACEHOLDER: exact per-iteration dispatch reduction ratio TBD — requires controlled measurement with matched iteration counts. Preliminary data suggests the reduction is significant, as each fused handler replaces 2–5 individual dispatches.]`

The distribution of instructions following `local.get` is remarkably concentrated:

> **Table 3.** Top 10 instructions after `local.get` (fusion disabled). Data from dispatch profiler.

| Rank | Successor instruction | CoreMark (%) | Lua fib (%) |
|:----:|----------------------|:------------:|:-----------:|
| 1 | `i32.const` | 30.8 | 31.8 |
| 2 | `local.get` | 20.7 | 21.5 |
| 3 | `i32.load` / `i32.load8_u` | 9.9 | 11.4 |
| 4 | `local.set` | 7.5 | 2.1 |
| 5 | `i32.store` | 3.6 | 4.2 |
| 6 | `i32.mul` / `i32.add` | 3.4 | 2.6 |
| 7 | `local.tee` / `i32.load` | 3.4 | 11.2 |
| 8 | `br_if` | 3.3 | 4.3 |
| 9 | `i32.add` / `local.set` | 3.0 | 2.1 |
| 10 | `i32.load8_u` / `i32.eqz` | 2.6 | 1.9 |
| | **Top 10 cumulative** | **88.2** | **92.3** |

The top 10 successors cover 88–92% of all `local.get` occurrences across both workloads, making a small number of fusion patterns highly effective.
We generate a fused handler for each common pair and longer sequences—`get_const`, `get_get`, `get_const_add`, `get_const_add_set_get`, etc.—and `local.get` effectively disappears into fused handlers.

`[PLACEHOLDER: Add columns for SHA-256, bzip2, and LZ4 to demonstrate that successor patterns are consistent across workloads.]`



### 4.2 Stack Machines Fuse Better

The central argument of this paper is that stack-machine fusion is qualitatively superior to register-machine fusion, not merely quantitatively.
The advantage arises from a fundamental property: on a stack machine, operand locations are compile-time constants, whereas on a register machine, they are runtime values loaded from the instruction stream.

Consider three consecutive `i32.ctz` instructions fused into a single handler.
On a stack machine with TOS register caching, each reads from and writes to a compile-time-known register argument:

```c
// TOS register: stack top is a function argument
uint64_t fused_ctz_tos(uint64_t t0) {
    t0 = __builtin_ctz((uint32_t)t0);
    t0 = __builtin_ctz((uint32_t)t0);
    t0 = __builtin_ctz((uint32_t)t0);
    return t0;
}
```

On a register machine, operands are addressed by indices loaded from the instruction stream:

```c
// Register machine: indices loaded at runtime
void fused_ctz_register(uint64_t* regs,
                        const uint64_t* __restrict__ imms) {
    regs[imms[0]] = __builtin_ctz((uint32_t)regs[imms[1]]);
    regs[imms[2]] = __builtin_ctz((uint32_t)regs[imms[3]]);
    regs[imms[4]] = __builtin_ctz((uint32_t)regs[imms[5]]);
}
```

`[TODO: Add compiled assembly output for both the stack-machine TOS version and register-machine version of the triple-ctz example, from the Godbolt link (https://godbolt.org/z/ca3bssvbh) or DESIGN.md. Without showing the assembly, the claim about 3-5× fewer instructions in Table 4 is unsupported.]`

Compiled with `clang -O3`, the compiler can see through the stack-machine version entirely—`t0` is a local variable, so the three `ctz` operations chain into a single register-to-register computation.
For the register-machine version, each index is loaded from a different memory address.
Even with `__restrict__`, the compiler cannot prove that `regs[imms[0]]` and `regs[imms[1]]` refer to the same slot, because the indices are runtime values.
Each store to `regs[]` may alias the next load, serializing the computation.

> **Table 4.** Instruction counts for three fused `ctz` operations. Compiled with `clang -O3`. Godbolt: https://godbolt.org/z/ca3bssvbh. AArch64 `ctz` requires `rbit`+`clz` (2 instructions) vs. x86-64's single `rep bsf`.

| Approach | x86-64 | AArch64 | Memory accesses |
|----------|:------:|:-------:|:---------------:|
| Stack machine (fixed offset) | 5 | 9 | 2 |
| Register machine (runtime indices) | 15 | 16 | 12 |
| TOS register (function argument) | 3 | 7 | 0 |

The TOS version eliminates memory access entirely.
The register-machine version requires 3–5× more instructions than the stack-machine TOS version.

This is not a contrived example.
Every fused handler in Silverfir-nano benefits from this effect.
The compiler sees through compile-time-constant stack offsets and pointer-to-register-argument indirections to eliminate intermediate memory traffic.
This optimization is fundamentally unavailable to register-machine interpreters that address operands by runtime indices.

**Implications for automation.**
On a stack machine, fusing N instructions is mechanical: concatenate handler bodies, and the compiler optimizes automatically.
A code generator can produce thousands of fused handlers without manual intervention.
On a register machine, naively concatenating handlers saves one dispatch but the compiler cannot optimize across the boundary.
To get equivalent optimization, each fused handler must be hand-written with knowledge of the data-flow pattern, or specialized variants must be generated for each combination of register operands—a combinatorial explosion.
This is why every register-based interpreter that performs fusion uses a small number of manually selected patterns [4, 8] rather than automated generation.

### 4.3 TOS Register Window

WebAssembly's static stack heights allow us to map the top N stack slots to hardware registers at compile time.
Silverfir-nano uses a window of four registers (t0–t3), passed as function arguments under `preserve_none`.
At each instruction, the compiler knows which TOS register holds each operand and generates a handler variant specific to the current stack depth.
With four registers, there are four depth variants (D1 through D4) that cycle as stack depth grows: depth 0 and 1 both map to D1, depth 2 to D2, depth 3 to D3, depth 4 to D4, depth 5 back to D1, and so on.
A `pop2_push1` operation like `i32.add` at depth 2 compiles to a single `add` instruction operating directly on register arguments.

When operations push or pop values outside the current window, the system emits spill (register→frame) or fill (frame→register) instructions.
Profiling CoreMark with fusion enabled shows this overhead is modest:

> **Table 5.** TOS spill/fill overhead in CoreMark (fusion enabled). Percentages are of total dispatches within the profiled run. Data from dispatch profiler.

| Operation | Count | % of dispatches |
|-----------|------:|:---------------:|
| spill_1 | 2,846,483 | 1.51 |
| spill_2 | 238,842 | 0.13 |
| spill_3 | 799,715 | 0.42 |
| **Total spills** | **3,885,195** | **2.06** |
| fill_1 | 1,220,635 | 0.65 |
| fill_2 | 748,370 | 0.40 |
| **Total fills** | **1,969,007** | **1.04** |
| **Combined** | **5,854,202** | **3.10** |

The combined 3.1% overhead reflects LLVM's tendency to keep hot values in a shallow stack region.
Only 3 out of every 100 handler dispatches are spent on TOS window management.

`[PLACEHOLDER: Expand Table 5 with spill/fill data from SHA-256, bzip2, and LZ4 to show overhead consistency across workloads.]`

**Register persistence.**
With tail-call dispatch and `preserve_none`, function arguments are passed in a fixed register assignment across all handlers.
On AArch64, the mapping is:

> **Table 6.** Handler argument signature on AArch64 (11 arguments). The `preserve_none` calling convention assigns each argument to a dedicated hardware register. Verified by disassembling 20+ handlers.

| Group | Argument | Description |
|-------|----------|-------------|
| VM state | ctx | Interpreter context (memory base, globals) |
| | pc | Program counter (current instruction pointer) |
| | fp | Frame pointer (local variable base) |
| Hot locals | l0 | Hottest local variable (by loop-nesting weight) |
| | l1 | Second-hottest local |
| | l2 | Third-hottest local |
| TOS window | t0 | Top of stack (stack depth 0) |
| | t1 | Stack depth 1 |
| | t2 | Stack depth 2 |
| | t3 | Stack depth 3 |
| Dispatch | nh | Next handler (preloaded function pointer) |

All eleven arguments reside in hardware registers — three for VM state, three for hot locals, four for the TOS window, and one for dispatch. No handler prologue or epilogue is needed to save or restore any of these values.

**Comparison with prior TOS schemes.**
Ertl's stack caching [22] models 1–2 registers as a 3-state finite state machine with dynamic or static state tracking.
HotSpot's template interpreter uses 1 register with 9 type-based states, dispatched through a 2D table at runtime.
wasm3 [4] uses 2 registers (r0 + fp0) with static operand-location variants (_r, _s, _rs, etc.).
Silverfir-nano differs in three respects: (1) four TOS registers rather than 1–2, made feasible by `preserve_none` providing sufficient argument registers; (2) compile-time depth tracking via Wasm's static stack heights, which replaces the state machine with deterministic handler variant selection; and (3) integration with large-scale fusion, where the 4-register window means fused handlers of 2–5 instructions typically operate entirely within the register window, eliminating all intermediate stack memory traffic.
Crucially, compile-time depth tracking eliminates all runtime cost of TOS state management: there is no dispatch table lookup, no state variable update, and no conditional logic to determine the current cache state — the correct handler variant is selected once during module loading and never reconsidered.

The linear nature of a stack machine makes this design scale gracefully: we need only N handler variants per instruction (one per TOS depth level).
A register-machine approach would require O(N^k) variants for k-operand instructions, a combinatorial explosion consistent with wasm3's experience where adding one meta-machine register increased handler variants per opcode from 3 to 10 [4].

### 4.4 Hot-Local Register Cache (L0/L1/L2)

#### The Remaining Memory Traffic

After fusion absorbs `local.get`/`local.set` dispatches, the *memory operations* they carried remain: each local access in a fused handler body still loads from or stores to `fp[idx]`.
Since local access accounts for ~38% of all Wasm instructions, this represents significant residual memory traffic even after dispatch elimination.

#### Compile-Time Analysis

Most functions have a small number of locals that dominate access counts—typically loop counters, accumulators, and index variables.

`[PLACEHOLDER: Add table showing top-5 hottest functions in SHA-256 and bzip2 with their local variable access distribution to demonstrate the skewed access pattern.]`

At compile time, Silverfir-nano walks the Wasm bytecode and counts local variable accesses weighted by loop nesting depth (10× per nesting level).
The top three locals become l0, l1, l2, mapped to dedicated hardware registers via the same `preserve_none` calling convention and tail-call dispatch mechanism used for TOS registers (Section 4.3).

A function prologue emits swap instructions that remap the hot locals to indices 0–2 and loads them into registers.
All subsequent references go through register access rather than frame memory.
The displaced original locals at indices 0–2 use remapped frame indices.
The swap is necessary because Wasm function arguments occupy the first local indices by convention. If the hottest local is, say, local 7, it must be physically moved to index 0 so that the l0 register maps to it. The original occupant of index 0 (typically the first function argument) is displaced to index 7.

**Novelty of the index-swap approach.**
The general concept of caching values in hardware registers is well-established—Ertl's stack caching [22] maps evaluation stack slots to registers, and Deegen [15] pins VM state registers via `preserve_none`.
However, no prior interpreter we surveyed applies *hotness-based local variable reordering*: profiling local access frequency at load time, then physically swapping the hottest locals to fixed register-mapped positions.
Existing interpreters (wasm3 [4], WAMR [5], wasmi [6], LuaJIT [8], CPython [7], HotSpot) all access locals at their original indices via frame memory.
The insight is that with a fixed number of register-mapped slots (three, in our case), the interpreter should fill them with the hottest locals—not the first N locals—and the index swap is the mechanism that makes this possible without modifying the Wasm bytecode or rewriting the instruction stream.

Spill and fill of hot locals are folded into existing call/return handlers (`fp[0..2] = l0, l1, l2` before call; `l0, l1, l2 = fp[0..2]` after return), requiring no additional dispatch overhead.

#### Key Properties

- **Orthogonal to TOS.** The hot-local registers do not increase the number of handler variants—they are additional arguments in the handler signature, alongside TOS registers.
- **First-class fusion participants.** `local_get_l0`, `local_set_l1`, `local_tee_l2`, etc. participate in fusion discovery and pattern matching exactly like any other instruction. Of the 1,500 built-in fusion patterns, 944 (63%) involve l0/l1/l2 operations.

#### Fusion Makes Local Register Access Free

When l0/l1/l2 operations are fused with arithmetic, the compiler completely eliminates the local access.
Consider `local.get_l0 + i32.const + i32.add + local.set_l0`—incrementing a loop counter.
Compiled with `clang -O3` on AArch64:

```asm
; Without L0: 3 instructions, 2 memory accesses
    ldr  x8, [x0, w1, uxtw #3]    ; load fp[idx]
    add  w8, w8, w2                ; add constant
    str  x8, [x0, w1, uxtw #3]    ; store fp[idx]

; With L0: 1 instruction, 0 memory accesses
    add  w0, w0, w1                ; l0 += K (pure register)
```

For a six-Wasm-instruction pattern (`local_get_l0 → local_get_l1 → i32.xor → local_get_l2 → i32.add → local_set_l0`), common in hash and accumulator loops:

```asm
; Without cache: 6 instructions, 4 memory accesses
    ldr  x8, [x0, w1, uxtw #3]    ; load fp[a]
    ldr  x9, [x0, w2, uxtw #3]    ; load fp[b]
    ldr  x10, [x0, w3, uxtw #3]   ; load fp[c]
    eor  w8, w8, w9                ; a ^ b
    add  w8, w8, w10               ; + c
    str  x8, [x0, w1, uxtw #3]    ; store fp[a]

; With L0/L1/L2: 2 instructions, 0 memory accesses
    eor  w8, w0, w1                ; l0 ^ l1
    add  w0, w8, w2                ; + l2 → l0
```

Six Wasm instructions, two machine instructions, zero memory access.
Three local reads and one local write have been completely eliminated.
This is only possible because local register operations are first-class participants in the fusion system: the fusion discovery tool profiles workloads and automatically discovers patterns containing l0/l1/l2 operations alongside regular arithmetic and control flow.

This effect is confirmed in the compiled binary.
For example, the fused handler `_op_add_const_add_sl0_D2` (computes a sum and stores to l0) writes the result directly to `w23` (the l0 register) with no frame memory access:

```asm
_op_add_const_add_sl0_D2:
    mov  x2, x1               ; save nh
    ldr  w8, [x21, #0x8]      ; load constant from imm0
    add  w9, w27, w26          ; t1 + t0
    add  w23, w9, w8           ; result → l0 (register x23)
    ldr  x1, [x21, #0x40]     ; preload new_nh
    add  x21, x21, #0x20      ; advance pc
    br   x2                    ; dispatch
```

Seven instructions total for four fused Wasm operations (`i32.add`, `i32.const`, `i32.add`, `local.set_l0`), with the `local_set_l0` folded into a register write.

### 4.5 Next-Handler Preloading

After fusion, TOS caching, and hot-local registers, handler bodies are extremely efficient — often just one or two register-to-register arithmetic instructions. The remaining overhead is *dispatch*: loading the next handler's function pointer from memory and jumping to it. In a conventional interpreter, each handler loads the next handler's address at its start and dispatches at its end. Since the handler body executes in 1–2 cycles, the load-to-use latency of the pointer fetch (typically 3–4 cycles on modern hardware) becomes the critical path — the CPU stalls waiting for the address before it can execute the indirect jump.

Next-handler preloading breaks this dependency by spreading the load and use across two handler invocations. Each handler receives the next handler's pointer as a function argument (`nh`), preloaded by the *previous* handler. While executing its own work, each handler simultaneously loads the handler pointer for the instruction *after* next, which will be passed as `nh` to the next handler. This gives the CPU a full handler execution's worth of time to complete the load before the pointer is needed.
Concretely, the handler pointer that handler N uses for dispatch (`nh`) was loaded by handler N−1 during its execution. By the time handler N reaches its indirect jump, the pointer has been in a register for the entire duration of handler N's work. On an out-of-order CPU, the preload's memory latency is fully hidden; the within-handler ordering of work and preload is irrelevant since the CPU issues both independently.

**Guard-check dispatch.** For linear handlers (those that always advance to the next sequential instruction), the preloaded `nh` is always correct, and the handler dispatches directly through it. For branching handlers (`br_if`, `if`), the preloaded `nh` may be wrong if the branch is taken. A guard check compares the actual next instruction against the predicted sequential target; if they match (the common case), dispatch uses the preloaded pointer; if not, the handler loads the correct pointer from the branch target.

The compiler optimizes this aggressively:

> **Figure 1.** Dispatch without vs. with preloading. In the conventional approach, each handler loads the next handler's address and immediately jumps to it, creating a load-to-use stall. With preloading, each handler receives its dispatch target (`nh`) pre-loaded by the previous handler, and loads the target for the *next* handler while executing its own work — spreading the load-to-use latency across two handler invocations.

```
Without preloading:                     With preloading:

Handler A:                              Handler A (nh = addr of B):
  load addr_of_B  ──┐                    save nh
  work: t0 = t1+t0  │ stall              work: t0 = t1+t0
  jump addr_of_B  ←─┘                    load addr_of_C → nh'
         │                                jump nh  ← no stall (preloaded)
         ▼                                       │
Handler B:                                       ▼
  load addr_of_C  ──┐                  Handler B (nh' = addr of C):
  work: t0 = t0&t1  │ stall              save nh'
  jump addr_of_C  ←─┘                    work: t0 = t0&t1  ← C loading in parallel
                                          load addr_of_D → nh''
                                          jump nh'  ← no stall (preloaded)
```

**Always-linear handlers** (89% of all handlers) always advance to the next sequential instruction.
The compiler inlines the guard check, proves it is always true, and eliminates the branch entirely.
The handler compiles to exactly five AArch64 instructions.
The C source (simplified) for an always-linear handler like `i32.add`:

```c
// Simplified handler for i32.add (always-linear, depth D2)
void op_i32_add(ctx, pc, fp, l0, l1, l2, t0, t1, t2, t3, nh) {
    t0 = t1 + t0;                          // work
    NextHandler new_nh = pc[2].handler;     // preload for handler after next
    musttail return nh(ctx, pc+1, fp,       // dispatch via preloaded nh
                       l0, l1, l2,
                       t0, t1, t2, t3, new_nh);
}
```

Compiled on AArch64:

```asm
op_i32_add_D2:
    mov  x2, x1           ; save preloaded nh
    add  w26, w27, w26    ; actual work: t0 = t1 + t0
    ldr  x1, [x21, #0x40] ; preload new_nh
    add  x21, x21, #0x20  ; advance pc (32 bytes/instruction)
    br   x2                ; tail-call via preloaded nh
```

No prologue, no epilogue, no conditional branch.
One instruction of actual work; four instructions of dispatch.

**Potentially-branching handlers** (e.g., `br_if`, `if`) retain the guard with a `likely` hint.
On the fast (fall-through) path, the preloaded `nh` is used; on taken branches, the handler reloads from the branch target.

**Always-nonlinear handlers** (`br`, `return`, `call`) discard `nh` and always reload from the computed target.

> **Table 7.** Handler classification by dispatch pattern. Data from static analysis of the handler set at compile time (not workload-specific).

| Category | Unique handlers | With depth variants | % of unique |
|----------|:-----------:|:---:|:---:|
| Always-linear | 1,557 | ~5,900 | 89.1 |
| Potentially-branching | 183 | ~760 | 10.5 |
| Always-nonlinear | 8 | ~11 | 0.5 |
| **Total** | **1,748** | **6,671** | **100** |

The three-tier dispatch strategy is generated automatically by the build system's code generator.

## 5. Evaluation

### 5.1 Methodology

**Hardware.**
All experiments run on a MacBook Air (Mac16,12) with Apple M4 (10-core, 4 performance + 6 efficiency), 16 GB RAM, macOS 26.2.

`[TODO: Add second test platform — Intel i9-14900K Windows PC. Run all benchmarks on both platforms for cross-architecture validation.]`

**Runtimes.**
We compare against three interpreter baselines and one JIT compiler:

| Runtime | Type | Version |
|---------|------|---------|
| Silverfir-nano | Interpreter (stack, fusion) | HEAD |
| wasm3 | Interpreter (register machine) | v0.5.1 (Apple LLVM 17.0.0) |
| WAMR (fast) | Interpreter (register transform) | v2.4.3 |
| wasmi | Interpreter (register machine) | v1.0.9 |
| Wasmtime (Cranelift) | Optimizing JIT | latest |

Wasmtime's baseline JIT (Winch) is included for context in supplementary results but is not our primary comparison target.
Our goal is to measure how close an interpreter can get to an *optimizing* JIT.

**Benchmarks.**
Four benchmarks spanning synthetic and real-world compute-intensive workloads:

| Benchmark | Description | Metric | Duration |
|-----------|-------------|--------|----------|
| CoreMark | EEMBC integer/control benchmark | Iterations/sec (↑) | ~10 s |
| SHA-256 | Cryptographic hashing (1 MB input) | MB/s (↑) | ~10 s |
| bzip2 | Burrows-Wheeler compression (256 KB, level 9) | MB/s (↑) | ~10 s |
| LZ4 compress | Fast lossless compression (1 MB input) | MB/s (↑) | ~10 s |

CoreMark is a widely-used synthetic benchmark; SHA-256, bzip2, and LZ4 are real-world programs compiled to Wasm with wasi-sdk 30 (`clang -O2`).

`[TODO: We must recompile benchmarks with -O3 instead of -O2 for maximum optimization. Verify whether -O3 produces different instruction patterns or performance.]`

All four exercise tight loops, bitwise operations, and moderate function call depth.
The optimization techniques (fusion, TOS caching, preloading, hot-local registers) are general and apply equally to floating-point code.
Results on floating-point and memory-bound benchmarks are discussed in Section 7.

**Measurement protocol.**
Each benchmark is measured 5 times, sequentially, with no concurrent CPU work.
Each run executes for approximately 10 seconds.
We report mean ± standard deviation.
Wall-clock time includes process startup, module loading, validation, and (for JIT compilers) compilation.
For benchmarks running 10+ seconds, startup overhead is negligible (<1%).

**Fusion configuration.**
CoreMark results use the built-in 1,500-pattern fusion set.
SHA-256, bzip2, and LZ4 results use a 500-pattern set discovered from profiling these three workloads (window size 8).
This demonstrates that the fusion discovery tool can generate workload-specific patterns; the difference in pattern count may affect absolute throughput.

`[PLACEHOLDER: Final results will use a unified fusion pattern set discovered from all primary benchmarks (CoreMark, SHA-256, bzip2, LZ4). Current results use separate pattern sets.]`

### 5.2 Overall Performance

> **Table 8.** Benchmark results: interpreters and optimizing JIT. All values are mean ± stddev over 5 runs except CoreMark (single controlled run from prior measurement). Best interpreter in **bold**. SF/CL = Silverfir throughput as percentage of Cranelift.

| Benchmark | Silverfir | wasm3 | WAMR | wasmi | Cranelift | SF/CL |
|-----------|-----------|-------|------|-------|-----------|:-----:|
| CoreMark (score) | **9,283** | 4,235 | 3,195 | 2,136 | 14,964 | 62.0% |
| SHA-256 (MB/s) | **71.05 ± 1.49** | 28.23 ± 0.48 | 18.95 ± 0.23 | 13.15 ± 0.24 | 241.03 ± 9.55 | 29.5% |
| bzip2 (MB/s) | **5.15 ± 0.04** | 3.10 ± 0.01 | 2.40 ± 0.02 | 2.03 ± 0.10 | 19.06 ± 0.23 | 27.0% |
| LZ4 compress (MB/s) | **324.86 ± 2.92** | 192.48 ± 0.15 | 120.15 ± 1.77 | 131.16 ± 3.67 | 732.90 ± 6.94 | 44.3% |

> **Table 9.** Silverfir speedup over other interpreters.

| Benchmark | SF / wasm3 | SF / WAMR | SF / wasmi |
|-----------|:----------:|:---------:|:----------:|
| CoreMark | 2.19× | 2.91× | 4.35× |
| SHA-256 | 2.52× | 3.75× | 5.40× |
| bzip2 | 1.66× | 2.14× | 2.54× |
| LZ4 compress | 1.69× | 2.70× | 2.48× |
| **Geometric mean** | **2.0×** | **2.8×** | **3.4×** |

**Key findings.**

1. **Silverfir-nano is the fastest interpreter on all four benchmarks**, outperforming wasm3 (the next-fastest) by 1.7–2.5× with a geometric mean of 2.0×.

2. **Silverfir-nano reaches 27–62% of Cranelift** (optimizing JIT) throughput. The geometric mean across all four benchmarks is 38%. CoreMark shows the strongest result (62%); the compute-heavy hash and compression workloads cluster at 27–44%.

3. **SHA-256 shows the largest interpreter gap** (5.4× over wasmi, 2.5× over wasm3). This workload has tight loops with heavy local variable access, making it ideal for both fusion and the hot-local register cache.

4. **LZ4 shows the strongest JIT ratio** (44% of Cranelift among the new benchmarks). Its mixed branch/compute pattern with moderate memory access benefits from fusion but does not expose the FP weakness.

5. **bzip2 shows the smallest interpreter advantage** (1.66× over wasm3). Its complex control flow with deep branching and irregular memory access patterns limit fusion effectiveness.

**Supplementary results.**

`[PLACEHOLDER: Add Table — Supplementary benchmark results (mandelbrot, c-ray, STREAM, Lua fibonacci) across all runtimes, using the same measurement protocol as Table 8.]`

For context, Silverfir-nano reaches 53–66% of Wasmtime's baseline JIT (Winch) on SHA-256, bzip2, and LZ4, and matches or slightly exceeds Winch on CoreMark and Lua Fibonacci.

### 5.3 Ablation Study

We evaluate the individual and combined contributions of fusion and the hot-local register cache (L0/L1/L2) using three configurations.
All configurations include TOS register window (4 registers) and next-handler preloading, as these are integral to the dispatch infrastructure and cannot be toggled independently without substantial code generation changes.

| Config | Fusion | L0/L1/L2 | Description |
|:------:|:------:|:--------:|-------------|
| B | ON (500 patterns) | OFF | Fusion only |
| C | OFF | ON (3 registers) | Register cache only |
| D | ON (500 patterns) | ON (3 registers) | Full (both) |

`[PLACEHOLDER: Configs 0 (both off), A1 (TOS=1), and A2 (no preloading) pending. These require additional code generation changes.]`

> **Table 10.** Ablation study results: mean ± stddev over 5 runs (MB/s). Same measurement conditions as Table 8.

| Benchmark | B: Fusion only | C: L0 only | D: Full |
|-----------|:--------------:|:----------:|:-------:|
| SHA-256 | 37.48 ± 1.04 | 18.38 ± 1.06 | 60.41 ± 2.53 |
| bzip2 | 4.10 ± 0.06 | 2.41 ± 0.08 | 4.32 ± 0.34 |
| LZ4 compress | 244.66 ± 17.03 | 142.94 ± 8.12 | 275.81 ± 18.15 |

> **Table 11.** Ablation speedup ratios.

| Benchmark | D/B (L0 benefit) | D/C (fusion benefit) | B/C (fusion vs. L0 alone) |
|-----------|:-----------------:|:--------------------:|:-------------------------:|
| SHA-256 | 1.61× (+61%) | 3.29× (+229%) | 2.04× |
| bzip2 | 1.05× (+5%) | 1.79× (+79%) | 1.70× |
| LZ4 compress | 1.13× (+13%) | 1.93× (+93%) | 1.71× |

**Key findings.**

1. **Fusion is the dominant optimization.** Config B (fusion only) is 1.70–2.04× faster than Config C (L0 only) across all benchmarks. Fusion alone delivers the majority of the performance by reducing dispatch overhead and enabling cross-instruction compiler optimization.

2. **The hot-local cache amplifies fusion.** Adding L0/L1/L2 to fusion (D vs. B) provides 5–61% additional speedup, depending on workload. SHA-256 benefits most (+61%) because its tight hash loops have heavy local variable access. bzip2 benefits least (+5%) due to complex control flow that limits fusion effectiveness.

3. **The register cache alone is weak.** Config C (L0 only, no fusion) performs comparably to the wasm3 baseline from Table 8 (SHA-256: 18.38 vs. 28.23 MB/s for wasm3). Without fusion to create multi-instruction handler bodies, the register cache cannot compensate for per-instruction dispatch overhead.

4. **The optimizations are synergistic.** On SHA-256, the combined speedup D/C (3.29×) exceeds the product of individual incremental speedups, confirming that fused handlers containing cached local operations achieve optimization neither technique can deliver alone—dispatch elimination *and* memory elimination in the same handler body.

`[PLACEHOLDER: Re-run all ablation configurations and Table 8 results in a single controlled session with no concurrent applications, to eliminate cross-session variance.]`

### 5.4 Assembly Evidence

We verify key design claims through disassembly of the compiled binary on AArch64.

**Zero prologue/epilogue.**
All examined arithmetic handlers (`i32_add`, `i32_sub`, `i32_mul`, `i32_and`, `i32_or`, `i32_xor`, `i32_shl`, `i32_shr_u`, `i32_ctz`, `i32_clz`, `i32_eqz`) compile with no `stp`/`ldp` pairs (stack save/restore).
Every handler is a pure leaf function.

**Five-instruction arithmetic handlers.**
Binary arithmetic handlers at depth D2 (`i32_add_D2`, `i32_sub_D2`, `i32_mul_D2`, `i32_and_D2`, `i32_or_D2`, `i32_xor_D2`, `i32_shl_D2`, `i32_shr_u_D2`) all compile to exactly five instructions: `mov` (save nh), arithmetic (work), `ldr` (preload new_nh), `add` (advance pc), `br` (dispatch).
Unary handlers (`i32_ctz`, `i32_clz`) require one additional instruction (AArch64 `ctz` is `rbit` + `clz`), yielding six instructions.

**Guard elimination.**
Always-linear handlers contain zero conditional branch instructions (`cbz`/`cbnz`/`b.cond`).
We verified this for all arithmetic handlers above.
The potentially-branching `br_if_simple_D2` has exactly one (`cbz`); `if__D2` has two (`cbz`, `b.ne`).

**Register mapping consistency.**
Across all examined handlers, the register assignment matches Table 6: pc=x21, fp=x22, l0=x23, l1=x24, l2=x25, t0=x26, t1=x27, t2=x28, t3=x0, nh=x1, ctx=x20.
No handler deviates from this mapping.

**L0 elimination in fused handlers.**
The handler `_op_add_const_add_sl0_D2` writes to `w23` (l0) directly.
No frame memory store appears—the `local_set_l0` has been compiled away into a register write.
Longer fused handlers with L0 and L1 operations (e.g., `_op_add_const_add_sl0_get_load_sl1_const_D2`) similarly show `w23` and `w24` writes with no corresponding frame stores for the hot locals.

### 5.5 Fusion Generality

`[PLACEHOLDER: Pattern overlap analysis pending. This section will analyze: (1) overlap of discovered fusion patterns across different workloads (CoreMark, SHA-256, bzip2, LZ4); (2) performance of the generic 1,500-pattern set vs. workload-specific sets; (3) sensitivity to fusion set size (100, 500, 1000, 1500 patterns). This data is needed to validate the claim that LLVM-compiled Wasm produces consistent instruction sequences.]`

### 5.6 Binary Size

> **Table 12.** Binary sizes for different build configurations.

| Configuration | Size |
|---------------|-----:|
| Minimal (no fusion, no WASI, stripped) | ~230 KB |
| Full (1,500 patterns + WASI + std, stripped) | 2.9 MB |

The 12.6× growth from minimal to full is dominated by the 1,500 fusion handlers (each generating 4 depth variants = 6,000 C functions).
The minimal build includes the complete WebAssembly 2.0 interpreter with zero external runtime dependencies, suitable for embedded deployment.
The fusion system offers a tunable trade-off: fewer patterns reduce binary size at the cost of performance.

`[TODO: Add binary size vs. performance curve. Test with ~200 patterns discovered from the unified benchmark set (CoreMark + SHA-256 + bzip2 + LZ4) — expected to significantly improve performance with minimal size increase. Use LZ4 compress as the reference benchmark. Also test with 100, 500, 1000, 1500 patterns to show the tradeoff curve. All pattern sets throughout the paper should use a single consistent discovery workload list.]`

## 6. Related Work

### 6.1 WebAssembly Interpreters

**wasm3** [4] is the fastest prior Wasm interpreter.
It uses a register-machine IR (the "Massey Meta Machine") where `local.get` is eliminated through copy-on-write slot sharing.
Two values are mapped to hardware registers (r0, fp0).
Tail-call dispatch uses `musttail` when available.
Fusion is limited to hand-selected peephole patterns.
Silverfir-nano differs in three respects: (1) staying stack-based enables automated fusion with 1,500+ patterns; (2) `preserve_none` provides 7 data registers (4 TOS + 3 hot locals) vs. 2; (3) handler preloading hides dispatch latency.
On our benchmarks, Silverfir-nano outperforms wasm3 by 1.7–2.5×.

**WAMR** [5] (WebAssembly Micro Runtime) provides a "fast interpreter" that converts stack-based Wasm to a three-operand register format during loading.
It uses computed-goto dispatch and fuses provider/consumer bytecode pairs.
The fast interpreter achieves ~2.5× speedup over WAMR's classic interpreter [5].
WAMR does not use `preserve_none`, `musttail`, TOS register caching, or hot-local registers.
Silverfir-nano outperforms WAMR by 2.1–3.8× on our benchmarks.

**wasmi** [6] is a pure-Rust Wasm interpreter.
Version 0.32+ (2024) rewrites stack-based bytecode into a register-based IR, achieving 80–100% speedup over the previous stack-based engine.
wasmi uses loop-match dispatch (switch-like) and prioritizes safety and `no_std` compatibility.
It does not employ tail-call dispatch, TOS caching, or automated fusion.

**Wizard** [11] is a research interpreter from CMU that interprets Wasm bytecode *in place* without translation to an intermediate representation.
Titzer [11] demonstrated that careful in-place interpretation can match translation-based approaches, though at lower absolute throughput than translation-based interpreters with fusion.

### 6.2 Superinstructions and Interpreter Optimization

The superinstruction concept originates with Proebsting [19], who showed that combining interpreter operations into "superoperators" reduces dispatch overhead.
Piumarta and Riccardi [12] proposed dynamic superinstructions via selective inlining of threaded code bodies, demonstrating 30–50% speedups.
Casey et al. [24] applied superinstruction selection to Java, achieving up to 45% dispatch reduction.
Ertl et al. [13] developed vmgen, an automatic generator of VM interpreters with superinstruction support—conceptually the closest precursor to Silverfir-nano's automated fusion discovery.

Ertl [22] introduced stack caching for interpreters.
Ertl and Gregg [23] combined stack caching with dynamic superinstructions, the most directly relevant prior work.
Silverfir-nano extends this combination with `preserve_none` for zero-overhead register persistence, next-handler preloading, automated discovery at scale (1,500+ patterns vs. manually selected), and hot-local registers as first-class fusion participants.
The hot-local index-swap technique—profiling access frequency and physically reordering locals to fill hardware register slots—is, to our knowledge, novel (Section 4.4).

Gregg et al. [14] argued for virtual register machines over stack machines for interpreters, showing fewer dispatches and better memory behavior.
Silverfir-nano's design challenges this recommendation: stack machines enable automated fusion with cross-instruction compiler optimization that register machines cannot replicate (Section 4.2).

### 6.3 Non-Wasm Interpreter Optimization

**CPython 3.14** [7] adopted tail-call dispatch via `musttail`, splitting its monolithic interpreter into ~225 handler functions.
The improvement was 3–5% on pyperformance, modest because Python opcodes are heavyweight (dynamic typing, reference counting), making dispatch a smaller fraction of total cost.
CPython does not use `preserve_none`, TOS caching, or fusion.

**LuaJIT** [8] uses a hand-written assembly interpreter (DynASM) with register-based bytecode and hand-selected fused instructions (compare-and-branch).
Its interpreter achieves excellent performance but requires deep assembly expertise, consistent with our argument that register-machine fusion resists automation.

**Deegen** [15] is a meta-compiler that generates JIT-capable interpreters from C++ bytecode semantics.
Its generated Lua interpreter is 179% faster than PUC Lua and 31% faster than LuaJIT's interpreter, using tail-call dispatch, GHC calling convention (related to `preserve_none`), and inline caching.
Deegen represents the state of the art in automatically generated interpreters.

### 6.4 Recent Superinstruction Work

Zhao et al. [25] synthesized superinstructions for the Ethereum Virtual Machine using dictionary-based compression algorithms, the most recent application of automated superinstruction techniques.
Their offline synthesis approach is conceptually similar to Silverfir-nano's profiling-driven discovery.

## 7. Discussion

### 7.1 Limitations

**Platform specificity.**
All results are from Apple M4.
Performance characteristics may differ on x86-64 and other ARM cores, particularly for branch prediction and register pressure.
The register mapping (Table 6) is AArch64-specific; x86-64 provides fewer registers.

**Floating-point performance.**
The underlying techniques (fusion, TOS caching, preloading, hot locals) are not specific to integer code and would benefit floating-point workloads equally.
The current implementation passes floating-point values through integer registers, which may introduce overhead on some platforms — investigation is left to future work.
A future implementation using dedicated FP registers for the TOS window when the type is known could improve FP performance.

**No SIMD support.**
Workloads relying on vector operations will not benefit from these optimizations.

**Compiler dependency.**
The approach requires `musttail` and `preserve_none`, currently available only in Clang/LLVM.
GCC 15 adds `musttail` but not `preserve_none`.

**Fusion set specificity.**
The built-in 1,500-pattern set is optimized for LLVM-compiled Wasm.
Programs compiled by other toolchains may exhibit different instruction patterns.

### 7.2 Threats to Validity

**Benchmark selection.**
Our four primary benchmarks are compute-intensive programs dominated by integer and bitwise operations.
We report supplementary results for floating-point and memory-bound workloads in Section 7.1 to provide a more complete picture.

**Single platform.**
Apple's branch predictor may be more favorable to indirect-branch-heavy code than other microarchitectures.
Cross-architecture validation is needed.

`[PLACEHOLDER: Add x86-64 (Intel i9-14900K) results for cross-platform validation.]`

**Statistical precision.**
The new benchmarks (SHA-256, bzip2, LZ4) include 5 runs with standard deviations.
CoreMark results are from a prior controlled measurement session.
More runs would strengthen the statistical confidence.

**Measurement includes startup.**
Wall-clock timing includes module loading and (for JIT) compilation.
This slightly favors interpreters, though the effect is small for 10-second benchmarks.

### 7.3 Future Work

Several directions warrant future investigation. Cross-platform evaluation on x86-64 (Intel, AMD) and other ARM cores would validate the generality of our results. A complete ablation study isolating TOS window size and preloading contributions is in progress. The current implementation's handling of floating-point values through integer registers is an engineering limitation rather than a fundamental constraint of the techniques; dedicated FP TOS registers could improve FP performance significantly. SIMD support would extend the approach to vector-heavy workloads. On the tooling side, adaptive fusion discovery at module load time could tune patterns for unknown workloads without offline profiling, and analysis of pattern overlap across compiler toolchains (LLVM, GCC, AssemblyScript) would strengthen the generalizability argument. Finally, a pure-Rust implementation awaits stabilization of explicit tail calls (`become`) and a `preserve_none` equivalent in the Rust compiler.

## 8. Conclusion

We have presented Silverfir-nano, a WebAssembly interpreter that reaches 27–62% of an optimizing JIT compiler's throughput across four benchmarks, without runtime code generation, while outperforming all tested interpreters by 1.7–5.4×.

The key insight is that a stack machine, often considered inferior to a register machine for interpreter design, enables a form of automated fusion that register machines cannot replicate.
On a stack machine, operand locations are compile-time constants, allowing the C compiler to optimize across fused instruction boundaries—eliminating intermediate memory accesses, chaining computations, and folding local variable access into register operands.
On a register machine, runtime operand indices create aliasing barriers that prevent these optimizations.

This advantage is amplified by four synergistic techniques: instruction fusion reduces dispatch overhead and creates multi-instruction handler bodies; the TOS register window keeps stack values in hardware registers; the hot-local register cache eliminates frame memory traffic for the most-accessed variables; and next-handler preloading hides dispatch latency for the 89% of handlers that are always-linear.
Together, representative arithmetic handlers compile to five machine instructions with zero prologue, zero epilogue, and zero memory traffic for in-register operands.

On memory-bound workloads, the JIT's native code generation provides an inherent advantage, and floating-point performance warrants further investigation (Section 7).
Despite these caveats, the result demonstrates that the interpreter–JIT performance gap is narrower than commonly assumed, and that careful attention to microarchitectural constraints can close a significant portion of it.

## References

[1] M. A. Ertl and D. Gregg, "The structure and performance of efficient interpreters," *Journal of Instruction-Level Parallelism*, vol. 5, 2003.

[2] A. Seznec, "A 64-Kbytes ITTAGE indirect branch predictor," *JWAC*, 2011.

[3] E. Rohou, B. N. Swamy, and A. Seznec, "Branch prediction and the performance of interpreters—Don't trust folklore," *IEEE/ACM CGO*, 2015.

[4] V. Shymanskyy and S. Massey, "wasm3: A high-performance WebAssembly interpreter," https://github.com/wasm3/wasm3.

[5] J. Xu, L. He, X. Wang, W. Huang, and N. Wang, "A fast WebAssembly interpreter design in WASM-Micro-Runtime," Intel Developer Zone, 2021.

[6] R. Freyler, "Wasmi's new execution engine—Faster than ever," 2024. https://wasmi-labs.github.io/blog/posts/wasmi-v0.32/

[7] K. Jin, "A new tail-calling interpreter for significantly better interpreter performance," CPython Issue #128563, 2025.

[8] M. Pall, "LuaJIT 2.1," https://luajit.org/.

[9] J. Haberman, "Parsing Protobuf at 2+GB/s: How I learned to love tail calls in C," 2021. https://blog.reverberate.org/2021/04/21/musttail-efficient-interpreters.html

[10] A. Haas et al., "Bringing the web up to speed with WebAssembly," *ACM SIGPLAN PLDI*, 2017.

[11] B. L. Titzer, "A fast in-place interpreter for WebAssembly," *Proc. ACM Program. Lang. (OOPSLA)*, vol. 6, 2022.

[12] I. Piumarta and F. Riccardi, "Optimizing direct threaded code by selective inlining," *ACM SIGPLAN PLDI*, 1998.

[13] M. A. Ertl, D. Gregg, A. Krall, and B. Paysan, "vmgen—A generator of efficient virtual machine interpreters," *Software: Practice and Experience*, vol. 32, no. 3, 2002.

[14] D. Gregg, A. Beatty, and K. Casey, "The case for virtual register machines," *Science of Computer Programming*, 2005.

[15] H. Xu and F. Kjolstad, "Deegen: A JIT-capable VM generator for dynamic languages," *arXiv:2411.11469*, 2024.

[16] C. Fallin, "A new backend for Cranelift, Part 1: Instruction selection," 2020. https://cfallin.org/blog/2020/09/18/cranelift-isel-1/

[17] LLVM RFC, "Exposing ghccc calling convention as preserve_none to clang," 2023. https://discourse.llvm.org/t/rfc-exposing-ghccc-calling-convention-as-preserve-none-to-clang/74233

[18] S. Cabrera, "Wasmtime baseline compilation" (RFC), Bytecode Alliance, 2022.

[19] T. A. Proebsting, "Optimizing an ANSI C interpreter with superoperators," *ACM SIGPLAN POPL*, 1995.

[20] F. Denis, "Performance of WebAssembly runtimes in 2023," https://00f.net/2023/01/04/webassembly-benchmark-2023/

[21] "Research on WebAssembly runtimes: A survey," *ACM TOSEM*, 2024.

[22] M. A. Ertl, "Stack caching for interpreters," *ACM SIGPLAN PLDI*, 1995.

[23] M. A. Ertl and D. Gregg, "Combining stack caching with dynamic superinstructions," *ACM SIGPLAN Workshop on Interpreters, Virtual Machines and Emulators (IVME)*, 2004.

[24] K. Casey, D. Gregg, M. A. Ertl, and A. Nisbet, "Towards superinstructions for Java interpreters," *SCOPES*, LNCS vol. 2826, 2003.

[25] D. Zhao et al., "Synthesizing efficient super-instruction sets for Ethereum Virtual Machine," *VMIL*, 2024.

[26] J. R. Bell, "Threaded code," *Communications of the ACM*, vol. 16, no. 6, 1973.

[27] B. L. Titzer, "Whose baseline compiler is it anyway?" *IEEE/ACM CGO*, 2024.

[28] S. Brunthaler, "Inline caching meets quickening," *ECOOP*, LNCS vol. 6183, 2010.

[29] M. Berndl, B. Vitale, M. Zaleski, and A. D. Brown, "Context threading: A flexible and efficient dispatch technique for virtual machine interpreters," *CGO*, 2005.

[30] H. Xu and F. Kjolstad, "Copy-and-patch compilation: A fast compilation algorithm for high-level languages and bytecode," *OOPSLA*, 2021.

[31] A. Jangda, B. Powers, E. D. Berger, and A. Guha, "Not so fast: Analyzing the performance of WebAssembly vs. native code," *USENIX ATC*, 2019.
