- Each section is parsed from the shared module payload, and after parsing the
  cursor position is compared against the precomputed section-end offset to
  detect a length mismatch.

- The trailing length check is skipped for custom sections.

## Moves

- 2024-01-28 (3a8b5fd6) replaced [[per-section-sub-readers]]: sub-section
  readers reset their position to zero and so cannot report a function body's
  absolute byte offset within the module; parsing from one reader and tracking
  positions preserves the code_offset needed to address disassembled
  instructions (diff).

- 2024-03-13 (055cac01) replaced by [[section-parsing]]: position tracking let
  a section parser over-read into the following section's bytes and only
  detected it afterward by comparing positions (and exempted custom sections
  from the check), whereas splitting each section into its own bounded
  sub-payload makes over-read structurally impossible and checks every section
  uniformly (diff).
