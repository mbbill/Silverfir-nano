---
commit: babd2890
---
Measured on CoreMark: the predecoded-enum interpreter peaked at ~269 it/s,
still behind the in-place baseline. The plan doc's findings diagnose why:
large FastOp variants (BrTable carrying a Vec) inflate the enum size and the
code vector's I-cache footprint; offset-preserving Nops for block/loop inflate
dispatch count; memory/table/global ops borrow RefCell per op. Net effect, in
the author's words at the time: "We removed LEB decode but added larger IR and
more per-op work, so fast.rs currently loses to inplace.rs on real workloads."
Predecoding alone does not pay if the IR gets bigger — compact bytecode's
cache locality beats fat decoded enums.
