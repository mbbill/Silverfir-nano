- A WASI filestat's inode is derived from OS metadata, making all paths to one file
  agree: Unix uses `st_ino`, Windows uses volume-serial + file-index, and a
  host-path hash is used only as a non-Unix/-Windows fallback
  (`derive_ino_from_meta`).

## Moves

- 2025-08-08 (1b0bc613) replaced [[path-hash-inode]]: hashing the host path
  string gave a different inode to every path, so the same file reached through
  fd_filestat_get, path_filestat_get and fd_readdir, or through a hardlink,
  never agreed on inode; deriving from OS metadata (Unix st_ino, Windows
  volume-serial+file-index, path-hash only as a non-Unix fallback) makes inodes
  stable and consistent across all paths to one file (diff).
