---
commit: b01092f1
---
The gate's first form: the official testsuite vendored into the repo
(d7da4668), converted offline by the external wast2json tool, driven by a
JSON-reading harness (spectest_json.rs). The current design — .wast parsed
in-tree, suite downloaded and version-pinned by a build script, no offline
conversion — replaced this at the June 2025 resurrection.
