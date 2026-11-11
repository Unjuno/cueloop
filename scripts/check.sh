#!/bin/bash
# One-command production check: lint, both test runners, and the demo.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

echo "== ruff =="
"$PYTHON" -m ruff check src tests scripts/*.py

echo "== pytest =="
"$PYTHON" -m pytest -q

echo "== unittest discover =="
"$PYTHON" -m unittest discover -s tests 2>&1 | grep -E "^(OK|FAILED|Ran)"

echo "== demo replay =="
CUELOOP_TTS=0 scripts/demo.sh replay

echo "checks passed"
