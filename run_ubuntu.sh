#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is not installed. See README.md for installation instructions.' >&2
  exit 1
fi
exec uv run --python 3.12 python dichotic_editor.py "$@"
