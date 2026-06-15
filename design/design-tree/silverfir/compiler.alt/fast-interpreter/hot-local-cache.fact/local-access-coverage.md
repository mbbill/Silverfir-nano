2026-02-18 (author)

The hot-local register cache is sized small and bounded because local-access
counts concentrate sharply in a few functions and a few locals, with a
diminishing marginal return per cached slot.

On CoreMark, the top-3 functions account for 96% of all local accesses:

| Func | % of total | top-1 local (%) | top-2 locals (%) | # locals used |
|------|-----------|-----------------|-------------------|---------------|
| #6   | 44.7%     | local[0] 18.9%  | +local[3] 33.8%   | 23            |
| #5   | 26.6%     | local[2] 25.2%  | +local[4] 49.7%   | 19            |
| #10  | 24.4%     | local[3] 26.0%  | +local[2] 47.9%   | 10            |

Weighted-optimal register-cache coverage across all functions (118.1M weighted
local accesses):

- optimal l0: 26.1M / 118.1M = 22.1% of accesses
- optimal l1: 22.5M / 118.1M = 19.0% marginal (l0+l1 = 41.1%)
- remaining: 69.5M / 118.1M = 58.9% still generic

The second cached slot adds only 19.0% marginal coverage and ~58.9% of accesses
stay generic regardless of cache depth, which is why the cache is capped at the
three hottest locals (l0/l1/l2). This bears on register-budget sizing for any
future hot-local cache.
