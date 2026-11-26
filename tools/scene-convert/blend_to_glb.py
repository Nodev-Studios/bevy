"""A .blend scene (Cycles materials) to .glb for gfx_viewer.

    blender --background --factory-startup --python tools/scene-convert/blend_to_glb.py -- \
        --input main.blend --output Scene.glb

What it does, in order (each loss is printed as LOST, see common.py):
1. Opens the file; deletes what its render would not show; makes particle instances real objects.
2. Rewrites each material as one Principled BSDF with the source's images (common.simplify_material).
3. Turns area lamps into hemisphere spot lights (glTF has no area light).
4. Prints the world (HDRI / sky / colour) it cannot carry and the scene camera's pose.
5. Exports GLB: modifiers applied, lights, no cameras, no animation.
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

a = common.args()
src, out = a["--input"], a["--output"]
bpy.ops.wm.open_mainfile(filepath=src, load_ui=False)
bpy.ops.file.find_missing_files(directory=os.path.dirname(src))
common.log("NOTE", f"opened {src}, render engine {bpy.context.scene.render.engine}")
common.camera_pose()
common.keep_render_visible()
common.realize_particles()
common.simplify_materials()
common.area_lights_to_spots()
common.note_world()
common.photometric_emission()
common.export_glb(out)
