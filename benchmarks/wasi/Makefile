# Reproducible build for the WASI benchmark binaries.
#
# All benchmarks compile with the wasi-sdk clang to wasm32-wasip1 at -O2.
# Running `make` rebuilds every .wasm from source with the exact same
# command, so a fresh checkout reproduces the committed binaries.
#
#   make            # build all benchmark wasm binaries
#   make sha256     # build one benchmark
#   make profile    # also build the *_profile.wasm dump/profiling variants
#   make clean      # remove generated wasm
#
# Override the compiler if wasi-sdk clang is not first on PATH:
#   make CC=~/wasi-sdk/bin/clang

# Plain `:=` (not `?=`) so make's built-in CC=cc default does not win;
# a command-line `make CC=...` still overrides this.
CC     := clang
TARGET := --target=wasm32-wasip1
CFLAGS := -O2 $(TARGET)

# Self-timing driver shared by the refit benchmarks.
COMMON := common/bench.h

WASM := sha256/sha256.wasm bzip2/bzip2.wasm lz4/lz4.wasm fib/fib.wasm \
        funcref/funcref.wasm mandelbrot/mandel.wasm stream/stream.wasm \
        coremark/coremark.wasm c-ray/c-ray.wasm sqlite/sqlite_bench.wasm

.PHONY: all profile clean sha256 bzip2 lz4 fib funcref mandel stream coremark c-ray sqlite

all: $(WASM)

# ---- individual benchmarks --------------------------------------------

sha256: sha256/sha256.wasm
sha256/sha256.wasm: sha256/bench_sha256.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ sha256/bench_sha256.c

BZIP2_SRC := bzip2/bench_bzip2.c bzip2/blocksort.c bzip2/bzlib.c \
             bzip2/compress.c bzip2/crctable.c bzip2/decompress.c \
             bzip2/huffman.c bzip2/randtable.c
bzip2: bzip2/bzip2.wasm
bzip2/bzip2.wasm: $(BZIP2_SRC) $(COMMON)
	$(CC) $(CFLAGS) -o $@ $(BZIP2_SRC)

lz4: lz4/lz4.wasm
lz4/lz4.wasm: lz4/bench_lz4.c lz4/lz4.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ lz4/bench_lz4.c lz4/lz4.c

fib: fib/fib.wasm
fib/fib.wasm: fib/fib.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ fib/fib.c

funcref: funcref/funcref.wasm
funcref/funcref.wasm: funcref/funcref.c $(COMMON)
	$(CC) $(CFLAGS) -Wl,--export-table -Wl,--strip-debug \
	    -o $@ funcref/funcref.c

mandel: mandelbrot/mandel.wasm
mandelbrot/mandel.wasm: mandelbrot/mandel.c
	$(CC) $(CFLAGS) -o $@ mandelbrot/mandel.c

stream: stream/stream.wasm
stream/stream.wasm: stream/stream.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ stream/stream.c

# c-ray needs -D__unix__ to pick its gettimeofday-based timer; WASI defines
# neither __unix__ nor _WIN32, so the stock source #errors without it.
c-ray: c-ray/c-ray.wasm
c-ray/c-ray.wasm: c-ray/c-ray-f.c $(COMMON)
	$(CC) $(CFLAGS) -D__unix__ -o $@ c-ray/c-ray-f.c -lm

sqlite: sqlite/sqlite_bench.wasm
sqlite/sqlite_bench.wasm: sqlite/bench_sqlite.c \
             sqlite/sqlite-amalgamation-3470200/sqlite3.c $(COMMON)
	$(CC) $(CFLAGS) -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION \
	    -o $@ sqlite/bench_sqlite.c \
	    sqlite/sqlite-amalgamation-3470200/sqlite3.c

# CoreMark, built against its POSIX port (in coremark/port/). PERFORMANCE_RUN
# selects the standard seed set whose CRCs the suite validates. FLAGS_STR is
# required by the reporting rules and must name the actual flags used.
CM_SRC := coremark/core_main.c coremark/core_list_join.c \
          coremark/core_matrix.c coremark/core_state.c coremark/core_util.c \
          coremark/port/core_portme.c
# Newer wasi-sdk sysroots carry DWARF in their libc objects even without -g.
# It has no bearing on the score and needlessly triples the committed module.
CM_FLAGS := -O2 $(TARGET) -Icoremark -Icoremark/port -DPERFORMANCE_RUN=1 \
            -Wl,--strip-debug

coremark: coremark/coremark.wasm
coremark/coremark.wasm: $(CM_SRC) $(COMMON)
	$(CC) $(CM_FLAGS) -DFLAGS_STR='"-O2 $(TARGET)"' -o $@ $(CM_SRC)

# ---- profiling / native-dump variants ---------------------------------
# These use clock() for their own timing, which on WASI needs the
# emulated-process-clocks shim.

PROF_CFLAGS := $(CFLAGS) -D_WASI_EMULATED_PROCESS_CLOCKS -lwasi-emulated-process-clocks

.PHONY: profile
profile: sha256/sha256_profile.wasm bzip2/bzip2_profile.wasm lz4/lz4_profile.wasm

sha256/sha256_profile.wasm: sha256/bench_sha256_profile.c
	$(CC) $(PROF_CFLAGS) -o $@ sha256/bench_sha256_profile.c

bzip2/bzip2_profile.wasm: bzip2/bench_bzip2_profile.c \
             bzip2/blocksort.c bzip2/bzlib.c bzip2/compress.c \
             bzip2/crctable.c bzip2/decompress.c bzip2/huffman.c bzip2/randtable.c
	$(CC) $(PROF_CFLAGS) -o $@ $^

lz4/lz4_profile.wasm: lz4/bench_lz4_profile.c lz4/lz4.c
	$(CC) $(PROF_CFLAGS) -o $@ lz4/bench_lz4_profile.c lz4/lz4.c

clean:
	rm -f sha256/sha256.wasm bzip2/bzip2.wasm lz4/lz4.wasm fib/fib.wasm \
	      funcref/funcref.wasm mandelbrot/mandel.wasm stream/stream.wasm \
	      coremark/coremark.wasm c-ray/c-ray.wasm sqlite/sqlite_bench.wasm \
	      sha256/sha256_profile.wasm bzip2/bzip2_profile.wasm lz4/lz4_profile.wasm
