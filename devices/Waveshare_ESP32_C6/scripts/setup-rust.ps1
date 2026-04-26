param()

$ErrorActionPreference = "Stop"

rustup target add riscv32imac-unknown-none-elf
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

rustup component add rust-src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo install cargo-espflash espflash --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
