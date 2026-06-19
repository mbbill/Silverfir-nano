- A profile-fast CLI command runs the fast interpreter with the sequence profiler
  enabled and prints a frequency-ranked table of the top-N fuseable instruction
  windows.

- Fusion candidates are ranked by reduction potential ((sequence_length - 1) *
  frequency) over a fixed window size, with overlapping sub/super-sequences counted
  independently.

- The author reads the printed ranking and manually transcribes selected sequences
  into the handler spec's [fused] entries, hand-writing each entry's name,
  encoding fields, and TOS pattern.

## Moves

- 2026-02-06 (fbbce862) replaced by [[fusion]]: the profiler only printed a
  flat frequency ranking of fuseable windows (scored length-1 times frequency)
  that the author then read and hand-coded into `[[fused]]` patterns, which
  double-counted overlapping sequences and could not reason about prefix
  relationships; the discover-fusion command builds a pattern trie that
  captures exact counts for every N-gram prefix from one max-window run and
  greedily selects a globally non-redundant set, subtracting each chosen
  pattern's count from all its strict prefixes (prefix-overlap adjustment) and
  enforcing the 192-bit (3x64) immediate-encoding budget, then auto-generates
  the complete `[[fused]]` TOML entry (name, encoding fields, TOS pattern)
  instead of a human transcribing the ranking (code).
