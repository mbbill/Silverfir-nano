commit: 9e801234

The in-house LEB128 module was seeded from Mohanson's `leb128` project (MIT,
credited in the file header) — its extensive test-case tables were vendored
to pin the new decoder against a known-good implementation. The live decoder
at this point is still the simple loop-and-shift form, but the file carries a
commented-out C reference for a fully unrolled, branch-per-byte
`__stream_read_vu64_unchecked` — the stated target the in-house path exists to
reach. Vendoring in place (rather than depending on the crate) is what makes
that rewrite possible without a fork.
