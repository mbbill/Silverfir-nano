#include <stdio.h>
#include "../common/bench.h"

/* volatile arg so clang can't constant-fold fib(N) at -O2 */
static volatile int fib_arg = 27;

static int fib(int n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

static volatile int sink;
static void fib_batch(long n, void *ctx) {
    (void)ctx;
    int base = fib_arg; /* 27, loaded from volatile */
    long acc = 0;
    /* Alternate the argument so fib() is not loop-invariant and cannot be
     * hoisted out of the loop; accumulate so the calls can't be dropped. */
    for (long i = 0; i < n; i++) acc += fib(base - (int)(i & 1));
    sink = (int)acc;
}

int main(int argc, char **argv) {
    /* Validation: fixed small case, independent of the timed workload. */
    printf("fib(20) = %d\n", fib(20));
    bench_result result = bench_run(fib_batch, 0, argc, argv);
    if (result.n == 0) return 2;
    printf("fib: rate = %.2f fib27/s\n", result.rate);
    return 0;
}
