#!/usr/bin/env bash
# Downloads public scenes for tools/gfx-viewer.sh into a folder OUTSIDE this repository, converts the ones that
# ship as FBX into glTF (Blender, headless only), and prints each one's licence. Nothing here is ever committed:
# the scenes keep their own licences, and some need credit. gfx_viewer lists them from tools/scenes.ron.
#
#   sponza          Crytek Sponza, the Khronos glTF-Sample-Assets edition. CRYENGINE Limited License Agreement,
#                   NOT a Creative Commons licence: fine to look at here, never to ship or publish.
#   abeautifulgame  "A Beautiful Game", a chess set. CC-BY 4.0 (credit the authors named in its README.md).
#   bistro          Amazon Lumberyard Bistro, exterior and interior (with wine), from NVIDIA ORCA (FBX + DDS,
#                   890 MB, converted to bistro/BistroExterior.glb and bistro/BistroInterior_Wine.glb).
#                   CC-BY 4.0: credit Amazon Lumberyard.
#   pica-pica       SEED's PICA PICA mini diorama and robot, as Bevy's solari example loads them (glb, from
#                   bevyengine/bevy_asset_files at a fixed commit). CC-BY-NC 4.0: NOT for commercial use.
#   zero-day        Beeple's Zero Day (Measure One), from NVIDIA ORCA (FBX + DDS, 1.1 GB, converted to
#                   zero-day/ZeroDay.glb). CC-BY 4.0: credit Mike Winkelmann (Beeple).
# Bevy's own assets/models/FlightHelmet (CC0) is listed by gfx_viewer without any download.
#
# The FBX scenes are converted by tools/orca-fbx-to-gltf.py through `blender --background --factory-startup`
# (never a window); set $BLENDER if blender is not on PATH. What the conversion loses is written at the top of
# that script. A scene already converted is not downloaded or converted again (delete its folder to redo it).
#
# Usage: tools/fetch-scenes.sh [sponza|abeautifulgame|bistro|pica-pica|zero-day ...]   (no argument: all)
# Folder: $BEVY_GFX_SDK_DIR/scenes, default ~/.local/share/bevy-graficos/scenes (gfx_viewer's default --scenes-dir)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
dir="${BEVY_GFX_SDK_DIR:-$HOME/.local/share/bevy-graficos}/scenes"
blender="${BLENDER:-blender}"
[[ $# -gt 0 ]] || set -- sponza abeautifulgame bistro pica-pica zero-day
mkdir -p "$dir"

# Prints a licence file, indented.
show() { sed -e 's/^/  | /' "$1"; echo; }

# Downloads $1 to $2 unless it is there, through a partial file so a cut download is never taken as whole.
download() {
  local url="$1" out="$2"
  [[ -s "$out" ]] && return 0
  mkdir -p "$(dirname "$out")"
  echo "  downloading $url"
  curl -fL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

# Converts FBX $1 to GLB $2 with emissive strength $3.
convert() {
  local fbx="$1" glb="$2" emissive="$3"
  [[ -s "$glb" ]] && { echo "  $glb is there"; return 0; }
  command -v "$blender" >/dev/null || { echo "fetch-scenes: needs Blender to convert $fbx (set \$BLENDER)" >&2; exit 1; }
  echo "  converting $(basename "$fbx") (Blender, headless)"
  "$blender" --background --factory-startup --python "$here/orca-fbx-to-gltf.py" -- \
    --fbx "$fbx" --out "$glb.part.glb" --emissive-strength "$emissive" | grep -E '^orca-fbx-to-gltf' || true
  [[ -s "$glb.part.glb" ]] || { echo "fetch-scenes: the conversion of $fbx failed" >&2; exit 1; }
  mv "$glb.part.glb" "$glb"
}

khronos() {
  local path="$1" repo="$dir/glTF-Sample-Assets"
  if [[ ! -d "$repo/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/KhronosGroup/glTF-Sample-Assets "$repo"
  fi
  # Only the plain glTF form of each model, plus its licence and readme; earlier models stay checked out.
  git -C "$repo" sparse-checkout add "/$path/glTF/" "/$path/LICENSE.md" "/$path/README.md"
  git -C "$repo" checkout --quiet
  find "$repo/$path" \( -name '*.gltf' -o -name '*.glb' \) -printf '  %p\n'
  if [[ -f "$repo/$path/LICENSE.md" ]]; then
    show "$repo/$path/LICENSE.md"
    grep -iE 'credit|author|copyright|©' "$repo/$path/README.md" 2>/dev/null | head -5 | sed -e 's/^/  | README: /' || true
  else
    echo "  no LICENSE.md: read $repo/$path/README.md before using it"
  fi
}

for scene in "$@"; do
  echo
  echo "== $scene =="
  case "$scene" in
    sponza) khronos Models/Sponza ;;
    abeautifulgame) khronos Models/ABeautifulGame ;;
    bistro)
      # NVIDIA's public link (no sign-in); it redirects to a signed download of Bistro_v5_2.zip.
      download https://developer.nvidia.com/bistro "$dir/bistro/download/Bistro_v5_2.zip"
      src="$dir/bistro/download/Bistro_v5_2"
      [[ -f "$src/BistroExterior.fbx" ]] || unzip -q -o "$dir/bistro/download/Bistro_v5_2.zip" -d "$dir/bistro/download"
      show "$src/README.txt"
      sed -n '1,3p' "$src/LICENSE.txt" | sed -e 's/^/  | LICENSE.txt: /'
      # Emissive strength 1000, as Falcor's BistroInterior_Wine.pyscene scales its emissives. to tune.
      convert "$src/BistroInterior_Wine.fbx" "$dir/bistro/BistroInterior_Wine.glb" 1000
      convert "$src/BistroExterior.fbx" "$dir/bistro/BistroExterior.glb" 1000
      ;;
    pica-pica)
      base=https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica
      download "$base/mini_diorama_01.glb" "$dir/pica-pica/mini_diorama_01.glb"
      download "$base/robot_01.glb" "$dir/pica-pica/robot_01.glb"
      # The licence is in SEED's repository's README (the asset files carry none of their own).
      download https://raw.githubusercontent.com/SEED-EA/pica-pica-assets/master/README.md "$dir/pica-pica/README.md"
      sed -n '/^## License/,$p' "$dir/pica-pica/README.md" > "$dir/pica-pica/LICENSE.md"
      show "$dir/pica-pica/LICENSE.md"
      ;;
    zero-day)
      download https://developer.nvidia.com/beeple "$dir/zero-day/download/ZeroDay_v1.zip"
      src="$dir/zero-day/download"
      [[ -f "$src/MEASURE_ONE/MEASURE_ONE.fbx" ]] || unzip -q -o "$src/ZeroDay_v1.zip" -d "$src"
      show "$src/LICENSE.txt"
      show "$src/README.txt"
      # Zero Day is lit by its emissive surfaces alone. to tune with the scene's exposure in tools/scenes.ron.
      convert "$src/MEASURE_ONE/MEASURE_ONE.fbx" "$dir/zero-day/ZeroDay.glb" 100
      ;;
    *) echo "fetch-scenes: no scene called $scene (sponza, abeautifulgame, bistro, pica-pica, zero-day)" >&2; exit 2 ;;
  esac
done
echo
echo "Scenes are in $dir; tools/scenes.ron lists them for gfx_viewer."
