# Publishing

Publish the embeddable core and its allocation helper together. CLI, benchmark,
spec/WASI harnesses and other development tools remain non-publishable.

Before a release:

1. Review public API changes under [the API policy](PUBLIC_API_POLICY.md).
2. Run correctness, supported feature/MSRV and no_std checks. Review performance
   failures explicitly; do not hide them with relaxed thresholds.
3. Set both package versions and the core's helper dependency version. From a
   clean release checkout, verify packaging without uploading:

   ```sh
   cargo publish --dry-run --locked --registry crates-io \
     -p sf-nano-tracked-alloc -p sf-nano-core
   ```

4. Verify an independent consumer using only the unpacked packages, including
   JIT-only, interpreter-only and both engines. Check archive VCS provenance and
   contents after any release commit change.
5. Obtain explicit approval of the concrete packages and versions. API acceptance
   alone is not publication approval. Retain the accepted API with the release.

Publish the helper first, wait for registry availability, then verify and publish
core against it:

```sh
cargo publish --locked --registry crates-io -p sf-nano-tracked-alloc
cargo publish --dry-run --locked --registry crates-io -p sf-nano-core
cargo publish --locked --registry crates-io -p sf-nano-core
```

If a publish operation times out, check registry state before retrying.

For wasmi-benchmarks, change the adapter's core dependency from git/tag to the
released registry version, preserving its engine features. Apply API migration
changes to the adapter source and update its lockfile. Verify both engines with
no path/git overrides, then update Nano's pinned upstream revision and remove
any obsolete adapter migration patch. Full competitor benchmarks are not needed
for this integration check.
