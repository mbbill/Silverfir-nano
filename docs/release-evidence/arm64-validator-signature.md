# Validator inline control signatures

Native ARM64; six alternating before/after process pairs with nine checked samples per process. Both binaries use release LTO and identical immutable Wasm bytes. No concurrent builds or test runs. The safe constructor retains full validation.

| Module | Before (ms) | After (ms) | Elapsed change |
| --- | ---: | ---: | ---: |
| bz2 | 0.878638 | 0.849609 | -3.30% |
| pulldown-cmark | 1.961167 | 1.852750 | -5.53% |
| spidermonkey | 44.391416 | 42.469104 | -4.33% |
| ffmpeg | 171.347854 | 165.799563 | -3.24% |
| coremark | 0.088682 | 0.083589 | -5.74% |
| erc20 | 0.075201 | 0.074107 | -1.45% |
| argon2 | 0.345654 | 0.333915 | -3.40% |

These are Module::new measurements, not full instance startup or CI acceptance. Raw samples and binary/input/source hashes are in [the JSON evidence](arm64-validator-signature.json).
