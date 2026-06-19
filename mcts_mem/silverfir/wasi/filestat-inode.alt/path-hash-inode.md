- WASI filestat synthesizes a file's inode by hashing the host path string
  modulo 2^53, with a constant per-process device id (`synth_ino`).

## Facts

- 2025-08-08 (ab101dca) pitfall: the synthetic inode is masked to 53 bits in
  fd_filestat_get so guests that read st_ino through a double-precision float
  do not see it truncated (code).

## Moves

- 2025-08-08 (1b0bc613) replaced by [[filestat-inode]]: hashing the host path
  string gave a different inode to every path, so the same file reached through
  fd_filestat_get, path_filestat_get and fd_readdir, or through a hardlink,
  never agreed on inode; deriving from OS metadata (Unix st_ino, Windows
  volume-serial+file-index, path-hash only as a non-Unix fallback) makes inodes
  stable and consistent across all paths to one file (code).
