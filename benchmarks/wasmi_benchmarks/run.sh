#!/bin/zsh
# Collect a full wasmi-benchmarks run, one Criterion group per process.
#
#   WASMI_BENCH_REPO=/path/to/wasmi-benchmarks OUT=/path/to/output ./run.sh
#
# Running the whole suite in a single process is what upstream does, but one
# engine that aborts takes every remaining group with it (WAMR did, twice, for
# different reasons — see README.md). Per-group runs cost a few seconds of
# extra process startup and keep a failure local to one group. The per-group
# streams are concatenated into one cargo-criterion JSON stream at the end,
# which is what make_report.py consumes.
set -u

REPO=${WASMI_BENCH_REPO:-"$HOME/code/wasmi-benchmarks"}
OUT=${OUT:-"$PWD/wasmi-bench-run"}
FEATURES=${BENCH_FEATURES:-"interpreters,jits"}
GROUPS="$OUT/groups"

# Pin the host toolchain. A wasi-sdk (or any cross) clang earlier in PATH makes
# bindgen generate 32-bit layout assertions for WAMR, and the build fails with
# "index out of bounds" on every size assertion.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$HOME/.cargo/bin"
export CC=${CC:-/usr/bin/clang}
export CXX=${CXX:-/usr/bin/clang++}
# Fizzy uses std::basic_string_view<uint8_t> and builds with -Werror; current
# Apple libc++ deprecates char_traits<unsigned char>.
export CXXFLAGS="${CXXFLAGS:-} -Wno-error=deprecated-declarations"
export LIBCLANG_PATH=${LIBCLANG_PATH:-/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib}

# With every engine linked into one binary, WAMR overflows the default 8 MB
# main-thread stack on execute/counter-local.
ulimit -s 65520

mkdir -p "$GROUPS"
cd "$REPO" || exit 1
: > "$OUT/status"

echo "stage=build $(date -u +%FT%TZ)" >> "$OUT/status"
cargo bench --no-default-features --features "$FEATURES" \
    --bench criterion --no-run > "$OUT/build.log" 2>&1 || {
    echo "stage=build FAILED" >> "$OUT/status"; exit 1
}

# Ask the harness which groups exist rather than hard-coding case names: the
# suite adds cases and renames identifiers between revisions.
echo "stage=list $(date -u +%FT%TZ)" >> "$OUT/status"
cargo bench --no-default-features --features "$FEATURES" \
    --bench criterion -- --list > "$OUT/list.txt" 2>&1 || {
    echo "stage=list FAILED" >> "$OUT/status"; exit 1
}
grep -oE '^(execute|startup)/[^/]+' "$OUT/list.txt" | sort -u > "$OUT/groups.txt"
n=$(wc -l < "$OUT/groups.txt" | tr -d ' ')
echo "stage=run groups=$n $(date -u +%FT%TZ)" >> "$OUT/status"

rm -rf "$REPO/target/criterion"
i=0
while IFS= read -r g; do
    i=$((i + 1))
    f="$GROUPS/${g//\//_}.json"
    [ -s "$f" ] && continue          # resume: keep groups already collected
    echo "  [$i/$n] $g $(date -u +%FT%TZ)" >> "$OUT/status"
    cargo criterion --no-default-features --features "$FEATURES" \
        --bench criterion --message-format=json -- "$g" \
        > "$f" 2>> "$OUT/run.log"
    rc=$?
    [ $rc -ne 0 ] && echo "    rc=$rc $g" >> "$OUT/status"
done < "$OUT/groups.txt"

cat "$GROUPS"/*.json > "$OUT/criterion.json"
echo "stage=done $(date -u +%FT%TZ)" >> "$OUT/status"
echo "wrote $OUT/criterion.json"
