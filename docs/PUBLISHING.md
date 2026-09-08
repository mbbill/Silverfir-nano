# Publishing

Publish the embeddable core and its allocation helper together. CLI, benchmark,
spec/WASI harnesses and other development tools remain non-publishable.

Release identity is shared: both published crates inherit
`workspace.package.version`, and each new release uses the identical Git tag
(e.g. `0.6.0`). Historical tags such as `0.5` remain unchanged. Use a patch bump
for compatible fixes and, while below 1.0, a minor bump for incompatible API
changes. The helper follows the core's release version.

The lightweight `release version consistency` CI check validates package
inheritance, the helper dependency, Cargo.lock and README installation examples
on PRs and main; on numeric or v-prefixed tag pushes it also requires the tag to
match exactly. It does not publish packages or replace API review. Configure it
as a required PR check in the repository ruleset. A failed tag check cannot undo
a local `cargo publish`: the release procedure must check the tag before upload.

Use Python 3.11 or newer. To start the next version from a clean checkout:

```sh
python3 -m ci.release prepare 0.6.1
```

This updates the workspace version, helper dependency, lockfile and installation
examples together. The new version must exceed both the current version and all
local/remote release tags (including historical two-part tags). Network failures
stop the operation. Review and commit the diff, then open the release PR.
The script neither commits nor pushes nor uploads packages.

After the reviewed PR is merged, use a clean checkout of current remote main:

```sh
python3 -m ci.release tag
# Push the exact tag printed by the command, for example:
git push origin refs/tags/0.6.1
python3 -m ci.release check
```

The tag command derives the version from Cargo.toml and creates an annotated tag;
it never overwrites an existing tag. The check requires matching local and remote
tag objects pointing to HEAD, HEAD equal to current remote main, consistent
package versions and a clean worktree. Missing tags, stale remote state and
version rollback stop the process. Use `--remote NAME` before the subcommand if
the release remote is not origin. A new commit after tagging requires inspection,
not silently moving the release tag.

These checks establish source/version identity, not publication readiness. They
do not query crates.io versions or certify CI/API approval. Do not bypass them
with a manual upload; registry state must also be checked before any retry.

Before a release:


1. Review public API changes under [the API policy](PUBLIC_API_POLICY.md).
2. Run correctness, supported feature/MSRV and no_std checks. Review performance
   failures explicitly; do not hide them with relaxed thresholds.
3. Use the version preparation command above. From a clean release checkout,
   verify packaging without uploading:

   ```sh
   cargo publish --dry-run --locked --registry crates-io \
     -p sf-nano-tracked-alloc -p sf-nano-core
   ```

4. Inspect the actual archive's README, metadata, licenses and example commands.
   Render its public documentation and inspect the crate landing page. Remove
   draft/review placeholders, check links and scope feature claims to the tested
   engines and targets. Passing builds do not verify publication text.
5. Verify an independent consumer using only the unpacked packages, including
   JIT-only, interpreter-only and both engines. Check archive VCS provenance and
   contents after any release commit change.
6. Obtain explicit approval of the concrete packages and versions. API acceptance
   alone is not publication approval. Retain the accepted API with the release.

Use the tag command above, push the tag and wait for its version check to pass. Publish from that same clean tagged commit;
do not move a release tag to a different commit after publishing.

Publish the helper first, wait for registry availability, then verify and publish
core against it:

```sh
python3 -m ci.release check && \
  cargo publish --locked --registry crates-io -p sf-nano-tracked-alloc
cargo publish --dry-run --locked --registry crates-io -p sf-nano-core
python3 -m ci.release check && \
  cargo publish --locked --registry crates-io -p sf-nano-core
```

If a publish operation times out, check registry state before retrying.

For wasmi-benchmarks, change the adapter's core dependency from git/tag to the
released registry version, preserving its engine features. Apply API migration
changes to the adapter source and update its lockfile. Verify both engines with
no path/git overrides, then update Nano's pinned upstream revision and remove
any obsolete adapter migration patch. Full competitor benchmarks are not needed
for this integration check.
