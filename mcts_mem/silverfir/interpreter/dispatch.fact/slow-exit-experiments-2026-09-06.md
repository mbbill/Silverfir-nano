commit: 6a382ad6

The user authorized an independent interpreter investigation while JIT work continued, prioritizing generic improvements over benchmark-specific fusion. Neither candidate is in the selected JIT code.

Candidate 6a382ad6 gives memory.size for memory zero a native path using the entry-refreshed logical length. The handler is appended after the existing handler bank: the previous 339480-byte blob remains an identical prefix and only 88 bytes are appended. Placing the handler earlier shifted existing handlers and confounded timings. A local final comparison was contaminated by external compilation activity and is not acceptance evidence.

Native all20 run [34087297901](https://github.com/mbbill/Silverfir-nano/actions/runs/34087297901) gives geometric point estimates +0.15985% on AMD 9V74 and +0.20278% on N2, without conclusive target improvements at the family threshold. Negative rows include AMD mandelbrot -3.1875% and N2 fibonacci-tail -6.0838%; their helper classifications do not erase their numeric signs. Full rows and actual-job audits are in docs/x64-campaign-2026-09-06/interpreter-memory-size/. No wider native lead over wasmi is established.

Local-only candidate 4d144428 declines address fusion when the actual generated table has a native plain load but no native fused load; it adds no new handler bank or benchmark identity check. All 49069 compression and 10837 word_count narrow-load slow exits disappear, other eighteen modules' exit populations remain unchanged, and the handler blob stays byte-identical. Tests cover all six signed/unsigned narrow i64 loads, live high bits, bounds, wraparound, multiple memories and memory64; local ARM/x64 correctness, 175 interpreter spec files and cross-target compilation pass. There is no native timing result, so the candidate remains shelved. Removing dynamic slow exits identifies a mechanism, not its end-to-end benefit.

A one-time local Apple M1 Pro comparison against wasmi 2.0.0-beta.8 found a roughly 2.00245x geometric Nano throughput ratio across all20. This is a local anchor, not a Linux x64 ranking; it does not turn either unmerged candidate into a confirmed improvement.
