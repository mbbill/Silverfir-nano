commit: b59caeff

Measured call-boilerplate overhead (docs/ABI_PLAN.md §1) is the motivation for
the entry split. The old software-managed local-call ABI cost ~12 fixed ops per
call site and 6+2N ops per return on arm64, where LLVM uses ~3 per call / ~2 per
return. Categorising coremark func 6's 1006 SF instructions against LLVM's 433
shows the gap is dominated by memory traffic and call boilerplate, not missed
peepholes: memory loads 199 vs 54 (+145), stores 165 vs 26 (+139), reg-reg moves
160 vs 64 (+96), uncond branches 87 vs 23 (+64). Round-trip waste is ~13 ops per
call site over ~70 hot call sites (func 6/8/9), ~700 static instructions plus a
much larger dynamic count.

The two earlier patches in the same thread (MOVN encoding, Eqz->IntCompare
fusion) saved ~380 static instructions but barely moved coremark, because the
call boilerplate was absorbing the wins. Target: per-call-site overhead from ~12
down to <=5 native ops on arm64 (<=7 on x86_64), per-return from 6+2N down to
4+N. This is the kill-fact behind both the entry split and the eqz->IntCompare
re-decision.
