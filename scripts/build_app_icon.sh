#!/usr/bin/env bash
# Package the original PNG into every standard macOS icon representation.
set -euo pipefail
cd "$(dirname "$0")/.."

icon_source="src/eidory/ui/icons/app_icon.png"
icon_workdir="$(mktemp -d "${TMPDIR:-/tmp}/eidory-icon.XXXXXX")"
icon_set="$icon_workdir/Eidory.iconset"
mkdir -p "$icon_set"

for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$icon_source" --out "$icon_set/icon_${size}x${size}.png" >/dev/null
    retina_size=$((size * 2))
    sips -z "$retina_size" "$retina_size" "$icon_source" --out "$icon_set/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$icon_set" -o assets/app_icon.icns
echo "Built assets/app_icon.icns from $icon_source"
