"""A USD scene (ALab, NVIDIA's Attic) to .glb for gfx_viewer. NOT RUN YET: written for `fetch-scenes-extra.sh
alab|attic`, neither of which was converted (see the report); expect to adjust it on first use.

    blender --background --factory-startup --python tools/scene-convert/usd_to_glb.py -- \
        --input entry.usda --output Scene.glb

Blender's USD importer reads geometry, lights, cameras and UsdPreviewSurface materials (as Principled BSDFs).
MDL materials (Attic) and renderer-specific shaders (ALab's) are not read: those meshes come out grey and are
listed. Then the same clean-up as a .blend: area lights -> spot lights, materials simplified, no cameras.
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

a = common.args()
src, out = os.path.abspath(a["--input"]), a["--output"]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=src, import_materials=True, import_lights=True, import_cameras=True,
                      import_render=True, import_proxy=False, import_guide=False)
scene = bpy.context.scene
grey = [o.name for o in scene.objects if o.type == "MESH" and not any(s.material for s in o.material_slots)]
if grey:
    common.log("LOST", f"materials: {len(grey)} meshes have no readable material (MDL / renderer shaders), grey: {', '.join(grey[:6])}")
cameras = [o for o in scene.objects if o.type == "CAMERA"]
for cam in cameras:
    common.camera_pose(cam)
    bpy.data.objects.remove(cam, do_unlink=True)
common.keep_render_visible()
common.remove_volume_only()
common.simplify_materials()
common.area_lights_to_spots()
common.note_world()
common.export_glb(out)
