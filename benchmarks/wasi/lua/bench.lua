-- Shared self-timing driver for the Lua benchmarks.
--
-- The batch unit has fixed semantics. Calibration only chooses how many
-- identical units to repeat for the requested time; a separate fresh batch
-- produces the reported work/second rate.

local M = {}

local MIN_DT = 1e-3
local MAX_N = 2 ^ 28

function M.coarse()
   local value = os.clock()
   return not (value and value > 0)
end

function M.now()
   local value = os.clock()
   if value and value > 0 then return value end
   return os.time()
end

function M.align()
   if not M.coarse() then return M.now() end
   local start = os.time()
   local current = start
   while current == start do current = os.time() end
   return current
end

function M.target(default)
   if arg and #arg > 0 then
      local value = tonumber(arg[#arg])
      if value and value > 0 then return value end
   end
   return default or 2.0
end

function M.correctness_only()
   if not arg then return false end
   for i = 1, #arg do
      if arg[i] == "--bench-correctness" then return true end
   end
   return false
end

local function timed_batch(batch, n)
   local start = M.coarse() and M.align() or M.now()
   batch(n)
   return M.now() - start
end

function M.calibrate(batch, target)
   local probe_target = math.max(MIN_DT, target / 8)
   local n = 1

   if M.coarse() then
      target = math.max(target, 2.0)
      local start = M.align()
      local elapsed = 0
      local total = 0
      while elapsed < 1.0 and total < MAX_N do
         n = math.min(n, MAX_N - total)
         batch(n)
         total = total + n
         elapsed = M.now() - start
         if elapsed < 1.0 then n = math.min(MAX_N, n * 8) end
      end
      if elapsed <= 0 then return math.max(1, total) end
      return math.max(1, math.min(
         MAX_N, math.floor(total * target / elapsed + 0.5)))
   end

   while true do
      local elapsed = timed_batch(batch, n)
      if elapsed >= probe_target or n >= MAX_N then
         if elapsed <= 0 then return n end
         return math.max(1, math.min(
            MAX_N, math.floor(n * target / elapsed + 0.5)))
      end

      local next_n
      if elapsed < MIN_DT then
         next_n = math.min(MAX_N, n * 8)
      else
         next_n = math.floor(n * probe_target * 1.05 / elapsed)
         next_n = math.min(MAX_N, next_n, n * 8)
         if next_n <= n then next_n = n + 1 end
      end
      n = next_n
   end
end

function M.measure(batch, workload)
   local elapsed = timed_batch(batch, workload)
   if elapsed <= 0 then
      error("timer did not advance during measured batch")
   end
   return workload / elapsed, workload, elapsed
end

function M.run(batch, target)
   if M.correctness_only() then
      batch(1)
      print("BENCH_WORKLOAD=1 (correctness only)")
      return 1.0, 1, 1.0
   end
   local workload = M.calibrate(batch, target)
   print(string.format("BENCH_WORKLOAD=%d", workload))
   return M.measure(batch, workload)
end

return M
