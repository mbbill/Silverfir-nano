-- Shared self-timing driver for the Lua benchmarks.
--
-- Mirrors common/bench.h: a geometric ramp grows the batch until one batch
-- alone meets a wall-clock target, the clock is read once per BATCH, and the
-- reported metric is a rate taken from the final batch only. Total cost is
-- bounded at roughly 3x the target on any engine, so the same script costs
-- the same whether it runs on a native JIT or on an interpreter under qemu.
--
-- Same rule as the C driver: keep the batch unit small. The ramp cannot
-- regulate below the cost of one unit, so if one unit already exceeds the
-- target nothing can bound the run.
--
-- CLOCK RESOLUTION. os.clock() is a process-CPU clock, and not every WASI
-- runtime implements one -- wasmtime 47 returns 0.0, so we fall back to
-- os.time(), which counts whole seconds. A 1-second clock read naively
-- costs up to 1s of error, which at a 2s target is 50% -- enough to make a
-- cross-runtime comparison meaningless. The fix is to ALIGN the start of
-- the measurement to a tick edge (see M.align): the reading is then exact
-- to within one check interval rather than one whole second, so a coarse
-- clock still gives a trustworthy number at the default target.

local M = {}

-- True when os.clock() is unavailable and we are down to whole seconds.
function M.coarse()
   local c = os.clock()
   return not (c and c > 0)
end

function M.now()
   local c = os.clock()
   if c and c > 0 then return c end
   return os.time()
end

-- Busy-wait to the next tick edge and return the time there. On a fine
-- clock this is just "now" -- there is no edge worth waiting for.
function M.align()
   if not M.coarse() then return M.now() end
   local t0 = os.time()
   local t = t0
   while t == t0 do t = os.time() end
   return t
end

-- The target is the last command-line argument when it parses as a positive
-- number, matching the convention the C benchmarks use.
function M.target(default)
   if arg and #arg > 0 then
      local v = tonumber(arg[#arg])
      if v and v > 0 then return v end
   end
   return default or 2.0
end

-- Coarse-clock path: one aligned measurement, accumulating work until the
-- budget elapses. Per-batch timing is impossible at 1s resolution, so we
-- never try -- we count total work over one exactly-delimited interval.
local function ramp_coarse(batch, target)
   local t0 = M.align()
   local n, done, dt, sized = 1, 0, 0, false
   while dt < target do
      batch(n)
      done = done + n
      dt = M.now() - t0
      if dt == 0 then
         n = n * 2 -- still inside the first tick; nothing measurable yet
      elseif not sized then
         -- Rate is now roughly known: pick a batch that lets us notice the
         -- closing tick promptly (~20 checks across the budget).
         n = math.max(1, math.floor(done / dt / 20))
         sized = true
      end
   end
   return done / dt, done, dt
end

-- batch(n) must perform exactly n units of work.
-- Returns rate (units/second), the batch size, and its duration.
function M.ramp(batch, target)
   if M.coarse() then return ramp_coarse(batch, target) end

   local n, total = 1, 0
   while true do
      local t0 = M.now()
      batch(n)
      local dt = M.now() - t0
      total = total + dt

      -- Out of budget but the sample is still usable: take it rather than
      -- spend another, larger batch chasing the target exactly.
      local out_of_budget = total >= 3.0 * target and dt >= target / 8.0
      if dt >= target or out_of_budget then
         if dt <= 0 then dt = 1e-9 end
         return n / dt, n, dt
      end

      if dt < 1e-3 or dt < target / 8.0 then
         n = n * 8 -- too short to extrapolate from; grow blindly
      else
         local est = math.floor(n * target * 1.25 / dt)
         n = est > n and est or n + 1
      end
   end
end

return M
