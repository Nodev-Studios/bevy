"""An OBJ + MTL scene (McGuire Computer Graphics Archive: San Miguel) to .glb for gfx_viewer.

    blender --background --factory-startup --python tools/scene-convert/obj_to_glb.py -- \
        --input scene.obj --output Scene.glb [--keep-bump]

Blender's OBJ importer reads the MTL into Principled BSDFs (Kd/map_Kd colour, Ns -> roughness, map_d alpha, ...);
they are then simplified like any other (common.simplify_material): an alpha texture becomes a cut-out (MASK 0.5),
a constant d < 1 stays a blended alpha. The archive's MTL "bump" maps are height maps that its importer reads as
normal maps, which would be wrong: they are dropped unless --keep-bump. OBJ has no lights and no camera.
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

a = common.args()
src, out = os.path.abspath(a["--input"]), a["--output"]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=src, use_split_objects=True, use_split_groups=False)
if not a.get("--keep-bump"):
    dropped = 0
    for material in bpy.data.materials:
        if material.use_nodes and material.node_tree:
            for node in [n for n in material.node_tree.nodes if n.type in {"NORMAL_MAP", "BUMP"}]:
                material.node_tree.nodes.remove(node)
                dropped += 1
    if dropped:
        common.log("LOST", f"bump maps: {dropped} MTL bump (height) maps dropped (glTF has no bump)")
common.simplify_materials()
common.log("LOST", "lights: OBJ has none (the viewer adds its default sun and sky)")
common.export_glb(out)
