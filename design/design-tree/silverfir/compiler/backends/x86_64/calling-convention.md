- The x86_64 backend supports both the System V and Microsoft x64 (Windows) C
  calling conventions, collected into a `callconv/{sysv,win64}` module pair that
  export identical symbols; the backend imports one name set regardless of OS;
  the convention is chosen at build time by `target_os` cfg (`callconv`).

- On Windows targets it uses the Microsoft x64 C-ABI (RCX/RDX/R8/R9 integer
  args, RDI and RSI added to the callee-saved GP set) instead of System V.

## Facts

- 2026-03-25 (2d0838c7) pitfall: unlike System V (no callee-saved XMM
  registers), the Microsoft x64 ABI makes XMM6..XMM15 callee-saved and requires
  a 32-byte caller shadow space, so the Windows prologue must reserve shadow
  space plus 10 aligned XMM save slots and spill/reload XMM6..XMM15 around the
  function body, inflating the frame-padding computation (diff).

## Moves

- 2026-04-07 (11b835a2) replaced [[inline-cfg-callconv]]: the
  Win64-vs-System-V difference (callee-saved set, stack padding, XMM6-15
  prologue spills, trapping-trunc helper sequence) was expressed as
  `cfg(sf_os_windows)` arms scattered through abi/inst/enc/helpers, so the ABI
  choice had no single home; collecting both conventions into
  callconv/{sysv,win64} that export identical symbols makes the backend import
  one name set regardless of OS and adding an ABI variant a new submodule
  (diff).
