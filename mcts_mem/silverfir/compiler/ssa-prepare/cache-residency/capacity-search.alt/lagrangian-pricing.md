- Per-region dual prices charge each local's tree DP for the capacity units
  it consumes; the DP re-solves at fixed prices and a subgradient step raises
  a region's price where demand exceeds its capacity, damped by iteration
  count.

- A fixed number of price iterations runs per function, exposed as a policy
  parameter (`iters`, default 12).

- After the last iteration a feasibility projection extracts a
  capacity-respecting selection from the priced DP values.

## Moves

- 2026-07-12 (748c8416) replaced by [[capacity-search]]: the price iterations
  were statically neutral on arm64 (net −9 native instructions across the
  9-module corpus) and actively harmful in the register-scarce regime they
  were designed for (armv7 qemu coremark −3.99%), and their noise masked a
  tie-commitment flaw in feasible extraction; the price-free DP with
  potential-ordered knapsack extraction is smaller and no worse on every
  measured target (sourced)
