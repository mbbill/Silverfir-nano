local function fib(n)
  if n < 2 then return n end
  return fib(n-1) + fib(n-2)
end

local result = fib(34)
print(string.format("fib(34) = %d", result))
