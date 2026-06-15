- The x86_64 backend encodes the Win64-vs-System-V ABI difference (callee-saved
  GP set, stack padding, XMM6-15 prologue/epilogue spills) as
  `cfg(sf_os_windows)` arms embedded directly in REG_PLAN and the
  abi/inst/enc/helpers sources.

## Moves

- 2026-04-07 (11b835a2) replaced by [[calling-convention]]: the
  Win64-vs-System-V difference (callee-saved set, stack padding, XMM6-15
  prologue spills, trapping-trunc helper sequence) was expressed as
  `cfg(sf_os_windows)` arms scattered through abi/inst/enc/helpers, so the ABI
  choice had no single home; collecting both conventions into
  callconv/{sysv,win64} that export identical symbols makes the backend import
  one name set regardless of OS and adding an ABI variant a new submodule
  (diff).
