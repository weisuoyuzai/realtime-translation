#!/usr/bin/env bash
# Launch Live Translator (GUI). Pass --cli ... for headless mode.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating virtual environment ..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  if command -v nvidia-smi >/dev/null 2>&1; then .venv/bin/python -m pip install -r requirements-cuda.txt; fi
fi
exec .venv/bin/python -m live_translator "$@"
