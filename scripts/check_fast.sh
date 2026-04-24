#!/usr/bin/env bash
# Compatibility wrapper. Prefer: python3 scripts/check.py fast
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/check.py" fast "$@"
