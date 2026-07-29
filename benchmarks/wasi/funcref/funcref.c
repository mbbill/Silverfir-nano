#include <stdint.h>
#include <stdio.h>

#include "../common/bench.h"

typedef uint32_t (*step_fn)(uint32_t);

/*
 * Keep two distinct functions in the table so the volatile selector prevents
 * LLVM from devirtualizing the indirect call. The benchmark initializes the
 * selector to the first function; the second function exists only to keep the
 * target dynamic at compile time.
 */
__attribute__((noinline)) static uint32_t step_one(uint32_t value) {
    return value + 1;
}

__attribute__((noinline)) static uint32_t step_two(uint32_t value) {
    return value + 2;
}

static step_fn const table_targets[] = {step_one, step_two};
static volatile uint32_t table_selector;
static volatile uint32_t initial_value = 7;
static volatile uint32_t result_sink;

static uint32_t run_exported_table(long calls) {
    step_fn target = table_targets[table_selector];
    uint32_t value = initial_value;
    for (long i = 0; i < calls; i++) value = target(value);
    return value;
}

static uint32_t run_direct(long calls) {
    uint32_t value = initial_value;
    for (long i = 0; i < calls; i++) value = step_one(value);
    return value;
}

static void exported_table_batch(long calls, void *ctx) {
    (void)ctx;
    result_sink = run_exported_table(calls);
}

static void direct_batch(long calls, void *ctx) {
    (void)ctx;
    result_sink = run_direct(calls);
}

int main(int argc, char **argv) {
    const long validation_calls = 257;
    const uint32_t expected = initial_value + (uint32_t)validation_calls;
    const uint32_t table_value = run_exported_table(validation_calls);
    const uint32_t direct_value = run_direct(validation_calls);

    if (table_value != expected || direct_value != expected) {
        fprintf(
            stderr,
            "funcref validation failed: table=%u direct=%u expected=%u\n",
            table_value,
            direct_value,
            expected
        );
        return 2;
    }
    printf(
        "funcref validates: table=%u direct=%u\n",
        table_value,
        direct_value
    );

    bench_result table = bench_run(exported_table_batch, 0, argc, argv);
    bench_result direct = bench_run(direct_batch, 0, argc, argv);
    printf("funcref exported-table: rate = %.2f calls/s\n", table.rate);
    printf("funcref direct: rate = %.2f calls/s\n", direct.rate);
    return 0;
}
