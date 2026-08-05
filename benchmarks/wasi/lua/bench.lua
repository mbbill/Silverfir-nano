-- Shared self-timing driver for the Lua benchmarks.
--
-- The batch unit has fixed semantics. With a high-resolution clock,
-- calibration only chooses how many identical units to repeat for the
-- requested time; a separate fresh batch produces the reported rate.
--
-- Some WASI runtimes do not provide a process CPU clock, so os.clock() stays
-- at zero and the only fallback is whole-second os.time(). On that path, a
-- tick-aligned one-unit calibration derives a roughly 10 ms chunk. A separate
-- tick-aligned measurement checks time only after each chunk until an integer
-- number of seconds has elapsed. The final-tick overshoot is therefore about
-- one chunk instead of as much as one second.

local M = {}

local MIN_DT = 1e-3
local MAX_N = 2 ^ 28
local COARSE_CHUNKS_PER_TICK = 100
local CLOCK_PROBE_ITERS = 100000
local clock_is_coarse

function M.coarse()
   if clock_is_coarse == nil then
      local before = os.clock()
      local probe = 0
      for i = 1, CLOCK_PROBE_ITERS do probe = probe + i end
      local after = os.clock()
      clock_is_coarse = not (before and after and after > before)
   end
   return clock_is_coarse
end

function M.now()
   if not M.coarse() then return os.clock() end
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

local function coarse_chunk(batch)
   -- Start immediately after a clock transition, then use the smallest batch
   -- unit and check every time. Timer-call overhead makes this estimate
   -- conservative: the measured chunk can only be shorter than intended.
   local start = M.align()
   local current = start
   local units = 0
   while current == start and units < MAX_N do
      batch(1)
      units = units + 1
      current = os.time()
   end
   if current == start then
      error("coarse timer did not advance during chunk calibration")
   end

   local ticks = current - start
   local units_per_tick = units / ticks
   local chunk = math.floor(units_per_tick / COARSE_CHUNKS_PER_TICK)
   return math.max(1, chunk), units, ticks
end

local function coarse_measure(batch, target)
   local chunk, calibration_units, calibration_ticks = coarse_chunk(batch)

   -- Calibration and measurement intentionally use different tick windows.
   -- Once aligned, each time check follows enough work to keep its overhead
   -- low, while the last chunk bounds endpoint overshoot to roughly 1% of a
   -- second whenever one benchmark unit is small enough.
   local start = M.align()
   local current = start
   local workload = 0
   while current - start < target do
      batch(chunk)
      workload = workload + chunk
      current = os.time()
   end
   local elapsed = current - start
   if elapsed <= 0 then
      error("coarse timer did not advance during measurement")
   end
   return workload / elapsed, workload, elapsed,
      chunk, calibration_units, calibration_ticks
end

function M.calibrate(batch, target)
   if M.coarse() then
      error("coarse clocks require chunked measurement through bench.run")
   end
   local probe_target = math.max(MIN_DT, target / 8)
   local n = 1

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
   if M.coarse() then
      local effective_target = math.max(1, math.ceil(target))
      print(string.format(
         "BENCH_TIMER=coarse-os.time resolution=1s tick_aligned=yes requested_target=%.3fs effective_target=%ds",
         target, effective_target))
      local rate, workload, elapsed, chunk, calibration_units,
         calibration_ticks = coarse_measure(batch, effective_target)
      print(string.format(
         "BENCH_CHUNK=%d calibration_units=%d calibration_ticks=%d target_chunks_per_tick=%d",
         chunk, calibration_units, calibration_ticks,
         COARSE_CHUNKS_PER_TICK))
      print(string.format("BENCH_WORKLOAD=%d", workload))
      return rate, workload, elapsed
   end
   local workload = M.calibrate(batch, target)
   print(string.format("BENCH_WORKLOAD=%d", workload))
   return M.measure(batch, workload)
end

return M
