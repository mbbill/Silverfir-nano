- The call stack pairs a pre-allocated spill buffer (the shadow stack of spill
  slots) with a separate Vec holding per-frame metadata (`CallStack`, `Frame`).

- Each frame is a lightweight struct of return pc, frame base, caller frame base,
  caller stack top, result-slots pointer, result count, and borrowed module /
  function references carrying a Store-tied lifetime.

- The call stack lives behind a RefCell in the context; push allocates the frame's
  slots in the spill buffer and records a Frame, and pop returns the Frame and
  recomputes the caller's spill pointer.

## Moves

- 2026-02-07 (87f0ff12) replaced by [[call-return]]: the old call stack split each
  frame between a pre-allocated spill buffer (the spill slots) and a separate
  Vec<Frame> of borrowed-lifetime metadata structs reached behind a RefCell, so
  every push/pop touched two structures and the RefCell guarded the frame vector
  on the hot call path; collapsing both into one contiguous u64 buffer where each
  frame is FRAME_METADATA_SLOTS (5) metadata words at negative offsets from its
  spill pointer followed by its spill slots removes the Vec<Frame>, removes the
  RefCell, and keeps a frame's metadata and spill area cache-adjacent, with frames
  linked by a caller_frame_start word and the root frame marked by its metadata
  sentinel (code).
