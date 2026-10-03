#!/usr/bin/env bash
set -euo pipefail
CYBERFLY_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$CYBERFLY_SCRIPT_DIR/manage.py" start
