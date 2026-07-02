#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
if [ "$(uname)" = "Darwin" ]; then
  mkdir -p build/helpers
  xcrun swiftc tools/linetop_overlay_helper/EidoryOverlayHelper.swift -O -o build/helpers/EidoryOverlayHelper
fi
python -m PyInstaller --clean --noconfirm Eidory.spec

echo "Built dist/Eidory.app"
