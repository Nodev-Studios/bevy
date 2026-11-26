#!/usr/bin/env bash
# Downloads more public test scenes for tools/gfx-viewer.sh into a folder OUTSIDE this repository, prints each
# one's licence, and converts the ones that are not glTF to .glb with Blender, headless. Nothing here is ever
# committed: every scene keeps its own licence (several are non-commercial: look only, never ship).
#
#   classroom          Blender demo "Classroom" (.blend, Cycles)            CC0
#   barbershop         Blender demo "Barbershop Interior" (.blend, Cycles)   CC-BY 4.0, Blender Animation Studio
#   bathroom           Bitterli "Salle de bain" by nacimus (Mitsuba + OBJ)   licence printed from LICENSE.txt
#   living-room        Bitterli "The Modern Living Room" by Wig42 (Mitsuba)  licence printed from LICENSE.txt
#   intel-sponza       Intel Sponza: base + curtains, ivy, trees, candles    personal/educational use (credits_license.txt)
#                      (glTF already, not converted; their 3ds Max / FBX / USD copies are not unpacked)
#   sun-temple         Epic "Sun Temple" (NVIDIA ORCA, FBX)                  CC-BY-NC-SA 4.0 (LICENSE.txt)
#   emerald-square     NVIDIA "Emerald Square", day and dusk (ORCA, FBX)     CC-BY-NC-SA 3.0 (LICENSE.txt)
#   san-miguel         McGuire Archive "San Miguel" (OBJ)                    research/education use with credit
#   khronos-extra      Khronos StainedGlassLamp (CC-BY 4.0), DragonAttenuation (Stanford, research use) (glTF)
#   alab               Netflix Animation Studios / Animal Logic "ALab" (USD) ASWF Digital Assets License
#                      (structure + 1k "techvar" assets, ~10.8 GB download; NOT in the default set, ask for it)
#   attic              NVIDIA "Attic" (USD + MDL): NOT fetched. Its only download needs a login. Put the zip
#                      Attic_NVIDIA*.zip into <scenes>/attic/download/ by hand and run this script with `attic`.
#
# Memory: every Blender conversion runs in its own systemd scope, capped at BLENDER_MEMORY without swap, so a scene
# too big kills only its own Blender. Emerald Square is converted in capped, resumable steps (scene-convert/bounded.py):
# if a step is killed, the script says which and what to change, and rerunning it resumes.
#
# Usage: tools/fetch-scenes-extra.sh [--no-convert] [--keep-archives] [ID ...]   (no ID: every default scene)
# Folder: $BEVY_GFX_SDK_DIR/scenes, default ~/.local/share/bevy-graficos/scenes (gfx_viewer's --scenes-dir).
# Each scene goes to <scenes>/<id>/: download/ (archives, deleted after a successful unpack unless
# --keep-archives), source/ (what was unpacked), and the .glb the viewer opens. Converting runs
#   blender --background --factory-startup --python tools/scene-convert/<script>.py -- ...
# (never a window). BLENDER overrides the binary (default: `blender` on PATH, else ~/.local/bin/blender).
# Every conversion writes <id>/convert.log with what it changed and what it lost.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
convert_dir="$here/scene-convert"
dir="${BEVY_GFX_SDK_DIR:-$HOME/.local/share/bevy-graficos}/scenes"
blender="${BLENDER:-$(command -v blender || echo "$HOME/.local/bin/blender")}"
convert=1
keep=0
ids=()
for arg in "$@"; do
  case "$arg" in
    --no-convert) convert=0 ;;
    --keep-archives) keep=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
    *) ids+=("$arg") ;;
  esac
done
[[ ${#ids[@]} -gt 0 ]] || ids=(classroom barbershop bathroom living-room khronos-extra sun-temple emerald-square
  san-miguel intel-sponza)

say() { printf '\n== %s ==\n' "$*"; }

# fetch URL FILE: resumable download of a public URL. No cookies, no credentials, a generic client name.
fetch() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" || -f "$(dirname "$(dirname "$out")")/source/.unpacked-$(basename "$out")" ]]; then
    echo "  have $(basename "$out")"; return
  fi
  echo "  downloading $(basename "$out") from $url"
  curl -fsSL --retry 3 --retry-delay 5 -C - -A "curl" -o "$out.part" "$url"
  mv "$out.part" "$out"
}

# unpack ARCHIVE DEST [SKIP_REGEX]: unzip/untar (a .zip's members matching the Python regex SKIP_REGEX are left
# out), then delete the archive unless --keep-archives. A marker file records a finished unpack.
unpack() {
  local archive="$1" dest="$2"; shift 2
  local mark="$dest/.unpacked-$(basename "$archive")"
  mkdir -p "$dest"
  if [[ ! -f "$mark" ]]; then
    echo "  unpacking $(basename "$archive")"
    case "$archive" in
      *.zip) python3 -I -c '
import re, sys, zipfile
archive, dest, skip = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
with zipfile.ZipFile(archive) as z:
    names = [n for n in z.namelist() if not (skip and re.search(skip, n)) and "/._" not in "/" + n]
    z.extractall(dest, names)
    print(f"  {len(names)} of {len(z.namelist())} files unpacked")' "$archive" "$dest" "$@" ;;
      *.tar.bz2|*.tbz2) tar -xjf "$archive" -C "$dest" --exclude='._*' ;;
      *.tar.gz|*.tgz) tar -xzf "$archive" -C "$dest" --exclude='._*' ;;
      *) echo "unpack: unknown archive $archive" >&2; return 1 ;;
    esac
    touch "$mark"
  fi
  if [[ $keep == 0 && -f "$archive" ]]; then rm -f "$archive"; fi
}

# licence FILE...: prints the licence files a download ships with (each the first 40 lines).
licence() {
  local found=0 file
  for file in "$@"; do
    [[ -f "$file" ]] || continue
    found=1
    echo "  licence file: $file"
    head -n 40 "$file" | tr -d '\r' | sed -e 's/^/  | /'
  done
  [[ $found == 1 ]] || echo "  no licence file in the download: the licence line above comes from the source's web page"
}

# run_blender ID SCRIPT ARGS...: one headless conversion, its output in <id>/convert.log.
run_blender() {
  local id="$1" script="$2"; shift 2
  [[ $convert == 1 ]] || { echo "  --no-convert: skipping $script"; return; }
  echo "  converting with $script (log: $dir/$id/${log_name:-convert.log})"
  local start=$SECONDS
  # A memory cap with no swap (BLENDER_MEMORY, default 12G): a scene too big to convert kills only Blender.
  systemd-run --user --scope --quiet -p MemoryMax="${BLENDER_MEMORY:-12G}" -p MemorySwapMax=0 \
    "$blender" --background --factory-startup --python-exit-code 1 \
    --python "$convert_dir/$script" -- "$@" >"$dir/$id/${log_name:-convert.log}" 2>&1 \
    || { echo "  conversion FAILED, see $dir/$id/${log_name:-convert.log}" >&2; tail -n 20 "$dir/$id/${log_name:-convert.log}" >&2; return 1; }
  grep -E '^(LOST|NOTE|WROTE|CAMERA|SUN)' "$dir/$id/${log_name:-convert.log}" | sort | uniq -c | sort -rn | head -n 30 | sed -e 's/^/  /' || true
  echo "  converted in $((SECONDS - start)) s"
}

# run_bounded ID SCRIPT OUT.gltf [BOUNDED OPTIONS] -- SCRIPT ARGS...: a conversion split into memory-capped Blender
# steps (scene-convert/bounded.py): each step is capped at BLENDER_MEMORY (default 8G) without swap, a killed step
# says so and keeps what finished, and rerunning resumes. Log: <id>/convert-<OUT stem>.log.
run_bounded() {
  local id="$1" script="$2" out="$3"; shift 3
  [[ $convert == 1 ]] || { echo "  --no-convert: skipping $script"; return; }
  echo "  converting with $script in capped steps (log: $dir/$id/convert-$(basename "${out%.gltf}").log)"
  local start=$SECONDS
  python3 -I "$convert_dir/bounded.py" --script "$script" --output "$out" --memory "${BLENDER_MEMORY:-8G}" \
    --blender "$blender" "$@" || return
  grep -E '^(LOST|NOTE)' "$dir/$id/convert-$(basename "${out%.gltf}").log" | sort | uniq -c | sort -rn | head -n 30 \
    | sed -e 's/^/  /' || true
  echo "  converted in $((SECONDS - start)) s"
}

scene_classroom() {
  local d="$dir/classroom"
  echo "  licence: CC0 (Christophe Seux, Blender demo files)"
  fetch https://download.blender.org/release/BlenderBenchmark2.0/scenes/classroom.tar.bz2 "$d/download/classroom.tar.bz2"
  unpack "$d/download/classroom.tar.bz2" "$d/source"
  licence "$d"/source/*/*.txt "$d"/source/*/LICENSE*
  run_blender classroom blend_to_glb.py --input "$d/source/classroom/main.blend" --output "$d/Classroom.glb"
}

scene_barbershop() {
  local d="$dir/barbershop"
  echo "  licence: CC-BY 4.0, credit Blender Animation Studio (Agent 327)"
  fetch https://download.blender.org/release/BlenderBenchmark2.0/scenes/barbershop_interior.tar.bz2 "$d/download/barbershop_interior.tar.bz2"
  unpack "$d/download/barbershop_interior.tar.bz2" "$d/source"
  licence "$d"/source/*/*.txt "$d"/source/*/LICENSE*
  run_blender barbershop blend_to_glb.py --input "$d/source/barbershop_interior/main.blend" --output "$d/Barbershop.glb"
}

# Bitterli's scenes: the Mitsuba 0.6 XML and its OBJ meshes; the materials are rebuilt from the XML's BSDFs.
bitterli() {
  local id="$1" zip="$2" out="$3" scale="${4:-1}"
  local d="$dir/$id"
  fetch "https://benedikt-bitterli.me/resources/mitsuba/$zip" "$d/download/$zip"
  unpack "$d/download/$zip" "$d/source"
  licence "$d"/source/*/LICENSE* "$d"/source/LICENSE* "$d"/source/*/*.txt
  run_blender "$id" mitsuba_to_glb.py --input "$(find "$d/source" -name 'scene_v0.6.xml' | head -n 1)" --output "$d/$out" --scale "$scale"
}
scene_bathroom() { bitterli bathroom bathroom2.zip Bathroom.glb 0.1; }  # modelled in decimetres
scene_living-room() { bitterli living-room living-room-3.zip LivingRoom.glb; }

scene_khronos-extra() {
  local d="$dir/khronos-extra" repo="$dir/khronos-extra/glTF-Sample-Assets"
  mkdir -p "$d"
  if [[ ! -d "$repo/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/KhronosGroup/glTF-Sample-Assets "$repo"
  fi
  git -C "$repo" sparse-checkout set --no-cone \
    /Models/StainedGlassLamp/glTF/ /Models/StainedGlassLamp/LICENSE.md /Models/StainedGlassLamp/README.md \
    /Models/DragonAttenuation/glTF/ /Models/DragonAttenuation/LICENSE.md /Models/DragonAttenuation/README.md
  git -C "$repo" checkout --quiet
  licence "$repo/Models/StainedGlassLamp/LICENSE.md" "$repo/Models/DragonAttenuation/LICENSE.md"
}

# orca ID NAME ZIP [FBX:OUT:EMISSIVE_SCALE ...]: an NVIDIA ORCA scene; each FBX of it becomes one glTF. The emissive
# scale is the one ORCA's own Falcor scene file (<scene>.pyscene, next to the FBX) applies. OUT.glb is converted in
# one Blender process; OUT.gltf goes through scene-convert/bounded.py (memory-capped steps, resumable: see there).
orca() {
  local id="$1" name="$2" zip="$3"; shift 3
  local d="$dir/$id" item fbx out scale input
  # developer.nvidia.com/<name> answers with a public, signed link to the zip; no login.
  fetch "https://developer.nvidia.com/$name" "$d/download/$zip"
  unpack "$d/download/$zip" "$d/source"
  licence $(find "$d/source" -maxdepth 3 -iname 'LICENSE*' | head -n 1)
  for item in "$@"; do
    IFS=: read -r fbx out scale <<<"$item"
    input="$(find "$d/source" -name "$fbx" | head -n 1)"
    if [[ $out == *.gltf ]]; then
      run_bounded "$id" fbx_to_glb.py "$d/$out" --orm-occlusion -- --orca --emissive-scale "$scale" --input "$input"
    else
      log_name="convert-${out%.glb}.log" run_blender "$id" fbx_to_glb.py --orca --emissive-scale "$scale" \
        --input "$input" --output "$d/$out"
    fi
  done
}
scene_sun-temple() { orca sun-temple sun-temple SunTemple_v4.zip SunTemple.fbx:SunTemple.glb:1000; }
scene_emerald-square() {
  # One Blender process for the whole block was killed by the OOM killer at 6.6 GB: converted in capped chunks.
  orca emerald-square emerald-square EmeraldSquare_v4_1.zip EmeraldSquare_Day.fbx:EmeraldSquareDay.gltf:1 \
    EmeraldSquare_Dusk.fbx:EmeraldSquareDusk.gltf:100  # x5 in Falcor; x100 to light the block in physical units (to tune)
}

scene_san-miguel() {
  local d="$dir/san-miguel"
  echo "  licence: free for research and educational use with attribution (Guillermo M. Leal Llaguno; license.txt)"
  fetch https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip "$d/download/San_Miguel.zip"
  unpack "$d/download/San_Miguel.zip" "$d/source"
  licence $(find "$d/source" -maxdepth 3 \( -iname '*licen*' -o -iname 'readme*' -o -iname '*.txt' \) | head -n 4)
  # The full model (~10 M triangles); if it does not convert within the memory cap, the archive's low-poly one.
  run_blender san-miguel obj_to_glb.py --input "$d/source/san-miguel.obj" --output "$d/SanMiguel.glb" \
    || { echo "  trying the low-poly version"; run_blender san-miguel obj_to_glb.py --input "$d/source/san-miguel-low-poly.obj" --output "$d/SanMiguelLow.glb"; }
}

# Intel's packages ship 3ds Max, FBX, USD and glTF copies of each scene; the 3ds Max, FBX and USD files and the
# preview renders are not unpacked.
scene_intel-sponza() {
  local d="$dir/intel-sponza" pkg
  for pkg in 830833:main_sponza 726650:pkg_a_curtains 726656:pkg_b_ivy1 726662:pkg_c_trees 726676:pkg_d_10k_candles; do
    local id="${pkg%%:*}" name="${pkg#*:}"
    if [[ -f "$d/source/.unpacked-$name.zip" ]]; then echo "  have $name"; continue; fi
    fetch "https://cdrdv2.intel.com/v1/dl/getContent/$id" "$d/download/$name.zip"
    unpack "$d/download/$name.zip" "$d/source" '(?i)\.(max|fbx|usd|usda|usdc|usdz)$|/Render_[^/]*\.(png|jpg)$'
  done
  licence "$d/source/main_sponza/credits_license.txt"
  # Its sun and lamps are at intensity 0 in the glTF: a lit copy beside it (the original is left as it is).
  python3 -I "$convert_dir/gltf_light_patch.py" --input "$d/source/main_sponza/NewSponza_Main_glTF_003.gltf" \
    --output "$d/source/main_sponza/NewSponza_Main_day.gltf" --sun-lux 20000 --drop-zero-lights | sed -e 's/^/  /'
  # And a night copy, every light removed, to pair with the 10 000 candles of pkg_d.
  python3 -I "$convert_dir/gltf_light_patch.py" --input "$d/source/main_sponza/NewSponza_Main_glTF_003.gltf" \
    --output "$d/source/main_sponza/NewSponza_Main_night.gltf" --drop-zero-lights | sed -e 's/^/  /'
  find "$d/source" -name '*.gltf' -printf '  %p\n'
}

# ALab: the USD structure (GitHub release) + the "techvar" assets (geometry, shaders, 1k textures). The 71 GB
# 4k texture pack and the 18 GB fur/cloth caches are not fetched.
scene_alab() {
  local d="$dir/alab"
  echo "  licence: ASWF Digital Assets License v1.0 (non-commercial, credit Netflix Animation Studios / Animal Logic)"
  fetch https://github.com/DigitalProductionExampleLibrary/ALab/archive/refs/tags/v2.3.0.zip "$d/download/alab-structure.v2.3.0.zip"
  unpack "$d/download/alab-structure.v2.3.0.zip" "$d/source"
  fetch https://dpel-assets.aswf.io/usd-alab/alab-techvars.v2.2.0.zip "$d/download/alab-techvars.v2.2.0.zip"
  unpack "$d/download/alab-techvars.v2.2.0.zip" "$d/source"
  licence $(find "$d/source" -maxdepth 3 -iname 'LICENSE*' | head -n 2)
  run_blender alab usd_to_glb.py --input "$(find "$d/source" -maxdepth 4 -name 'entry.usda' | head -n 1)" --output "$d/ALab.glb"
}

scene_attic() {
  local d="$dir/attic" zip
  zip="$(find "$d/download" -maxdepth 1 -iname 'Attic*.zip' 2>/dev/null | head -n 1)"
  if [[ -z "$zip" ]]; then
    echo "  skipped: NVIDIA's Attic download needs a login. Download Attic_NVIDIA.zip by hand from"
    echo "  https://developer.nvidia.com/usd (USD samples), put it in $d/download/ and run: $0 attic"
    return
  fi
  unpack "$zip" "$d/source"
  licence $(find "$d/source" -maxdepth 3 -iname '*licen*' | head -n 2)
  run_blender attic usd_to_glb.py --input "$(find "$d/source" -iname 'Attic_NVIDIA.usd*' | head -n 1)" --output "$d/Attic.glb"
}

for id in "${ids[@]}"; do
  if ! declare -F "scene_$id" >/dev/null; then
    echo "fetch-scenes-extra: no scene called $id" >&2; exit 2
  fi
  say "$id -> $dir/$id"
  "scene_$id"
done
echo
du -sh "$dir"/* 2>/dev/null | sed -e 's/^/  /'
