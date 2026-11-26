"""One chunk of a staged scene to a .gltf + .bin, for tools/scene-convert/bounded.py (which runs it; not run by hand).

    blender --background --factory-startup --python tools/scene-convert/export_chunk.py -- \
        --blend STAGE/scene.blend --objects CHUNK.json --output CHUNK_DIR/chunk.gltf

Appends only the chunk's objects (and the meshes, materials and images they use) from the staged .blend into an
empty file, so the memory used follows the chunk, not the scene. Textures are not decoded or re-encoded: the glTF
points at the image files the converter wrote (export_keep_originals).
"""

import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

a = common.args()
with open(a["--objects"]) as f:
    names = json.load(f)
bpy.ops.wm.read_factory_settings(use_empty=True)
with bpy.data.libraries.load(a["--blend"], link=False) as (src, dst):
    missing = sorted(set(names) - set(src.objects))
    dst.objects = [n for n in names if n in src.objects]
if missing:
    raise SystemExit(f"objects not in the staged scene: {missing[:5]}")
scene = bpy.context.scene
for obj in dst.objects:
    scene.collection.objects.link(obj)
out = a["--output"]
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=out,
    export_format="GLTF_SEPARATE",
    export_keep_originals=True,
    use_active_scene=True,
    export_apply=True,
    export_lights=True,
    export_cameras=False,
    export_extras=False,
    export_yup=True,
    export_materials="EXPORT",
    export_animations=False,
)
common.log("WROTE", f"{out} {len(dst.objects)} objects")
