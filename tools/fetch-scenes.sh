#!/usr/bin/env bash
# Downloads a few public glTF scenes for tools/gfx-viewer.sh into a folder OUTSIDE this repository, and prints
# each one's licence. Nothing here is ever committed: the scenes keep their own licences, and some need credit.
#
#   sponza          Crytek Sponza, the Khronos glTF-Sample-Assets edition: a courtyard with arches, cloth and
#                   plants. The classic test of bounce light and of a sun through an open roof. Its files are
#                   under the CRYENGINE Limited License Agreement, NOT a Creative Commons licence: fine to look
#                   at here, never to ship or publish.
#   abeautifulgame  "A Beautiful Game", a chess set on a board: glossy and rough materials, small shadows.
#                   CC-BY 4.0 (credit the authors named in its README.md).
# Bevy's own assets/models/FlightHelmet (CC0) is listed by gfx_viewer without any download.
#
# They come from https://github.com/KhronosGroup/glTF-Sample-Assets through a shallow, blobless, sparse
# checkout of only those folders, so only their files are downloaded. Each model's LICENSE.md and README.md in
# that repository state its licence; this script prints them after the download. Read them before using a scene
# anywhere but on your own screen.
#
# Usage: tools/fetch-scenes.sh [sponza|abeautifulgame ...]   (no argument: both)
# Folder: $BEVY_GFX_SDK_DIR/scenes, default ~/.local/share/bevy-graficos/scenes (gfx_viewer's default --scenes-dir)
set -euo pipefail

dir="${BEVY_GFX_SDK_DIR:-$HOME/.local/share/bevy-graficos}/scenes"
repo="$dir/glTF-Sample-Assets"
[[ $# -gt 0 ]] || set -- sponza abeautifulgame

paths=()
for scene in "$@"; do
  case "$scene" in
    sponza) paths+=(Models/Sponza) ;;
    abeautifulgame) paths+=(Models/ABeautifulGame) ;;
    *) echo "fetch-scenes: no scene called $scene (sponza, abeautifulgame)" >&2; exit 2 ;;
  esac
done

mkdir -p "$dir"
if [[ ! -d "$repo/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/KhronosGroup/glTF-Sample-Assets "$repo"
fi
# Only the plain glTF form of each model (not its other encodings or screenshots), plus its licence and readme.
git -C "$repo" sparse-checkout set --no-cone \
  $(for path in "${paths[@]}"; do
      printf '/%s/glTF/\n/%s/LICENSE.md\n/%s/README.md\n' "$path" "$path" "$path"
    done)
git -C "$repo" checkout --quiet

echo
echo "Scenes in $repo/Models (gfx_viewer lists every .gltf/.glb under $dir):"
for path in "${paths[@]}"; do
  echo
  echo "== ${path#Models/} =="
  find "$repo/$path" \( -name '*.gltf' -o -name '*.glb' \) -printf '  %p\n'
  if [[ -f "$repo/$path/LICENSE.md" ]]; then
    sed -e 's/^/  | /' "$repo/$path/LICENSE.md"
    grep -iE 'credit|author|copyright|©' "$repo/$path/README.md" 2>/dev/null | head -5 | sed -e 's/^/  | README: /' || true
  else
    echo "  no LICENSE.md: read $repo/$path/README.md before using it"
  fi
done
