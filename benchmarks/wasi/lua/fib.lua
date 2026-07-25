-- Fibonacci benchmark - exercises function calls, recursion, integer arithmetic
--
-- Self-timing: replaces the old fixed fib(38) / fib(34) pair, which took 26s
-- on the interpreter and needed a separate reduced workload for CI. The batch
-- unit is one fib(FIB_N) evaluation, small enough that a slow engine can still
-- regulate against the time target.
--
--   lua.wasm fib.lua [seconds]

local bench = dofile("bench.lua")

local function fib(n)
  if n < 2 then return n end
  return fib(n-1) + fib(n-2)
end

-- Validation: a fixed small case, independent of the timed workload.
print(string.format("fib(20) = %d", fib(20)))

local FIB_N = 20
local sink = 0

local function batch(n)
  local acc = 0
  for _ = 1, n do acc = acc + fib(FIB_N) end
  sink = acc
end

local rate = bench.ramp(batch, bench.target(2.0))
print(string.format("fib: rate = %.2f fib%d/s", rate, FIB_N))
