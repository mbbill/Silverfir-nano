---
commit: 467b9b16
---
FAST_INTERPRETER_PLAN.md ships in-repo with the FastIR bootstrap. Goals:
maximize single-threaded speed on stable Rust without JIT; correctness
preserved; feature-gated opt-in. Acceptance: spectest green in both modes,
CoreMark >1.5x baseline by Phase 3. The roadmap already names the future:
quickening (self-specializing ops resolving globals/tables/memory views to
pointers), superinstructions (peephole fusion of hot sequences), stack caching
(TOS/NOS), tight frames and RefCell-churn avoidance.
