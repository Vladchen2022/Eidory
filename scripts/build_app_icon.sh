#!/usr/bin/env bash
# Render the rounded icon and package every standard macOS representation.
set -euo pipefail
cd "$(dirname "$0")/.."

icon_source="src/eidory/ui/icons/app_icon.png"
# The 2048px original has generous empty margins. Keep the complete ribbon E,
# centered at about 68% of the tile width, without repainting its colors/faces.
xcrun swift tools/render_app_icon.swift assets/app_icon_source.png "$icon_source" 384 372 1280 1280
icon_workdir="$(mktemp -d "${TMPDIR:-/tmp}/eidory-icon.XXXXXX")"
trap 'rm -rf "$icon_workdir"' EXIT
icon_set="$icon_workdir/Eidory.iconset"
mkdir -p "$icon_set"

for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$icon_source" --out "$icon_set/icon_${size}x${size}.png" >/dev/null
    retina_size=$((size * 2))
    sips -z "$retina_size" "$retina_size" "$icon_source" --out "$icon_set/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$icon_set" -o assets/app_icon.icns
echo "Built assets/app_icon.icns from $icon_source"
