commit: b9b02d80

The typed banked residency redesign was motivated by the c-ray gap: under the untyped
single-bank prepare model, float values reenter the pipeline as untyped u64 and bounce
through GP registers (fmov x<->d shuffles) before the backend can use them as FP
operands. The proposal threaded exact Wasm value types through decode and prepare and
gave each bank its own budget so float values stay FP from prepared LIR through
MachineIR; the semantic stack stays one ordered typed stack and bank selection is a
property of value type only, not a second stack.

A first cut of the FP bank allowed a temporary GP-fallback (a float that did not fit
the FP transient bank silently fell back into a GP register); once the frontend carried
exact types and per-bank budgets enforced fit in prepare, that fallback was removed and
a float that does not fit became a hard lowering error, so ordinary float flow is never
represented as a GP MachineIR value.
