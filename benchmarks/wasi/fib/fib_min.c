// Freestanding fib: no libc, exports `fib` and a WASI-compatible `_start`.
// Build: clang -O2 -nostdlib -Wl,--export=fib -o fib_min.wasm fib_min.c
int fib(int n) {
    return n < 2 ? n : fib(n - 1) + fib(n - 2);
}

__attribute__((export_name("_start")))
void _start(void) {
    volatile int r = fib(30);
    (void)r;
}
