- The engine ships an in-tree WASI preview1 host implementation, gated as a
  host-side build feature (`sf_wasi_host`); WASI calls reach it through the
  runtime-call host path rather than being compiled into the guest, with no
  dependency on external WASI crates.

- The same preview1 host function table is registered under both the
  `wasi_snapshot_preview1` and the legacy `wasi_unstable` module namespaces
  (`wasi_imports`); the `wasi_unstable` registration reuses the identical
  implementations rather than a separate table.

- Each WASI call reaches guest linear memory through the calling module's first
  memory instance, reading and writing argument buffers there directly.

- A host context holds the host-side state (args, environment, preopens, the
  file descriptor table, and stdio-close tracking) behind shared interior
  mutability, threaded into every syscall implementation.

- Open files and directories live in a per-context fd table; guest fds are
  allocated sequentially above the preopens (3..3+N), while stdio (0/1/2) is
  handled as a special case outside the table.

- A guest path is resolved lexically under the preopen directory it is relative
  to: paths that are absolute or that climb above the preopen base with `..`
  are rejected with `NOTCAPABLE`, and per-fd rights are checked before each
  file operation; a guest cannot escape its granted directory
  (`resolve_under_base`).

- File identity (`ino`) is taken from whatever identity the host itself keeps
  for a file, and is synthesized from the path and its metadata only where the
  host offers none; the synthesized form is a fallback, never the primary
  answer on a host that can do better.

## Facts

- 2025-06-24 (0c6357bb) pitfall: import resolution must special-case the WASI
  module names — an import from `wasi_snapshot_preview1` / `wasi_unstable`
  with no registered host function fails fast with `Unlinkable` instead of
  falling through to the regular path, which would try to recursively
  instantiate the import's module name as if it were a Wasm module (code).

- 2025-08-07 (18a8526d) statement: native WASI took on the `filetime` crate as
  its one host dependency, used to implement the file-timestamp syscalls
  (fd_filestat_set_times / path_filestat_set_times) (code).

- 2025-08-07 (22df07db) pitfall: the original confinement check only treated a
  leading backslash as absolute, so a POSIX-style leading-'/' guest path
  slipped past on platforms where `Path::is_absolute` did not flag it;
  `resolve_under_base` now explicitly rejects both '/'- and '\'-leading paths,
  and guest path bytes are validated (NUL -> INVAL, non-UTF-8 -> ILSEQ) before
  resolution (code).

- 2025-10-26 rationale: fd_close on a stdio descriptor (fd 0/1/2) returns
  Success and only records the fd in a closed-stdio set; it never closes the
  host stdout/stderr stream, so a program that closes and reopens stdio is not
  broken by losing the real stream — the no-op is intentional and must not be
  "corrected" into an actual stream close (sourced).

- 2025-10-26 rationale: rights are monotonically non-increasing —
  fd_fdstat_set_rights returns `Notcapable` unless the requested base/inheriting
  rights are a subset of the fd's current rights, so a capability can never be
  escalated through this call; a rebuilder must preserve this rather than
  relaxing it into a free rights-set assignment (sourced).

- 2025-10-26 rationale: toggling the APPEND fdflag via fd_fdstat_set_flags
  reopens the underlying host file with `OpenOptions::append(true)` so the OS
  enforces atomic seek-to-end append, instead of emulating append in userspace
  by seeking to end on each write; a rebuilder could legitimately pick the
  userspace seek-to-end form, so the reopen is a recorded decision (sourced).

- 2026-06-14 statement: at HEAD nano no longer reopens the host file for APPEND —
  fd_fdstat_set_flags only stores the fdflags on the fd table entry, and fd_write
  emulates append in userspace by seeking to end (`SeekFrom::End(0)`) before each
  write when the stored APPEND flag is set; the author does not recall why the
  OS-reopen form was dropped, and the current userspace form works (sourced).

- 2025-10-26 rationale: reading from stdin (fd 0) is a deliberate non-feature —
  fd_read(0) writes nread=0 and returns Success (immediate EOF) rather than
  blocking on or reading host stdin, because most WASI programs do not consume
  stdin and supporting it would add host-IO and blocking complexity; a
  rebuilder must not "fix" this into a real stdin read (sourced).

- 2025-10-27 (8bb8bb89) rationale: proc_exit propagates its exit code by
  returning a `WasmError` whose `Exit` variant carries the code as a typed i32
  field (with an exit_code() accessor), not by formatting it into a message
  string; the prior string form forced the embedder to string-match the
  Display text to recover the code, which is fragile (code).

- 2025-10-30 (8d75a84e) rationale: the exit code is threaded structurally from
  proc_exit through the `WasmError::Exit` variant alone — the previously-parallel
  `WasiCtx.exit_code` field was a redundant second copy that nothing read and is
  removed, so the embedder recovers the code via `WasmError::exit_code()` rather
  than from host state or by string-parsing the message (code).

- 2026-02-20 (ff0461e3) rationale: fd_read on stdin slurps all remaining host
  stdin into a process-global static buffer on the first read and serves every
  later read from that buffer (tracking a position cursor), rather than reading
  the live stdin stream per call, to give the guest consistent/replayable read
  behavior across iovs and repeated invocations (code).

- 2026-06-14 rationale: real stdin is a necessity — without it basically nothing
  runs — so nano serves guest stdin reads from host stdin (slurped into a buffer),
  reversing the earlier immediate-EOF non-feature stance; a rebuilder must
  implement a real stdin read, not the EOF stub (sourced).

- 2026-04-16 (9ff58dcd) rationale: the file-times syscalls
  (fd_filestat_set_times / path_filestat_set_times) delegate to the external
  `filetime` crate (v0.2) rather than raw OS time-setting calls — atim/mtim
  nanoseconds are converted through ns_to_filetime and applied via
  set_file_handle_times / set_file_times / set_symlink_file_times. The crate is
  an engine-product dependency newly adopted here, declared optional and gated
  by the engine `wasi` feature (`wasi = ["dep:filetime"]`) (code).

- 2026-04-25 (da9a777a) pitfall: on RV32 Linux under qemu-riscv32-static, Rust
  std's metadata decode path (std::fs::metadata / symlink_metadata /
  File::metadata) faults or corrupts runtime-call state after a successful
  statx; the RV32/Linux WASI path issues the kernel statx ABI
  directly for path classification and filestat (`rv32_linux_stat`), and any
  removal of that direct path must be revalidated under qemu-riscv32-static
  (code).

- 2026-07-26 (003b6a21) rationale: identity has to come from the host because
  a hard link is two names for one file, and the WASI suite asserts the two
  report equal `ino`. Anything derived from the path answers that they are
  different files. Unix has the inode; Windows has a volume-relative file
  index, reachable only by asking the OS directly since
  `MetadataExt::file_index` is unstable (rust-lang/rust#63010) (code).

- 2026-07-26 (fe130e61) pitfall: a descriptor number inside the preopen range
  3..3+N is not necessarily still a preopen. `fd_renumber` retires the original
  and installs a live entry under that number, so an fd operation that decides
  by range arithmetic rather than by the preopen lookup -- which reports a
  retired preopen as absent -- answers BADF for a good descriptor. Every fd
  operation must resolve through the lookup (code).

## Moves

- 2025-08-07 (7d16c093) replaced [[wasmtime-wasi-backed]]: wasmtime-wasi/wiggle
  proved too complicated and heavy to carry — the attempt left wiggle's
  GuestMemory unimplemented and every generated binding body a todo!() — and nano
  does not need that complexity, so the external wasmtime-wasi/wiggle/witx deps
  were removed for the lean hand-rolled in-tree preview1 implementation (sourced).
