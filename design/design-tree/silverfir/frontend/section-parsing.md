- Each section is parsed from its own bounded sub-payload split off the module
  reader; over-reading into the following section is structurally impossible
  and the section-end check applies uniformly to every section (custom sections
  included).

- The sub-payload is carved out at a known absolute offset; a function
  body's byte offset within the module is preserved and addressable
  (`code_offset`) for later disassembly.

## Moves

- 2024-03-13 (055cac01) replaced [[position-tracking]]: position tracking let a
  section parser over-read into the following section's bytes and only detected
  it afterward by comparing positions (and exempted custom sections from the
  check), whereas splitting each section into its own bounded sub-payload makes
  over-read structurally impossible and checks every section uniformly (diff).
