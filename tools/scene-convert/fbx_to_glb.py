"""An FBX scene (NVIDIA ORCA: Sun Temple, Emerald Square) to .glb for gfx_viewer.

    blender --background --factory-startup --python tools/scene-convert/fbx_to_glb.py -- \
        --input Scene.fbx --output Scene.glb [--emissive-scale 1000] [--orca] [--camera NAME] [--skip-lights]

Geometry, lights and cameras come from Blender's FBX importer (FBX units -> metres).

--orca: the materials follow ORCA's texture convention (its README): <material>_BaseColor (RGB colour, A opacity),
_Specular (R occlusion, G roughness, B metalness: glTF's own ORM layout), _Normal (DirectX: green points down) and
_Emissive. Each material is rebuilt as one Principled BSDF from those files, found by name next to the FBX:
    base colour + a cut-out (alpha MASK 0.5) when the base colour's alpha has holes
    roughness = Specular.G, metalness = Specular.B (one metallicRoughness texture)
    normal map with its green channel flipped to glTF's convention (the converter writes a flipped copy)
    emission = the _Emissive texture x --emissive-scale (the scale ORCA's own Falcor scene files use)
Lost: ambient occlusion (Specular.R), the HDR environment maps, anything UE's materials did beyond the textures.
Without --orca, materials are simplified like a .blend's (common.simplify_material).
"""

import os
import re
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

a = common.args()
src, out = os.path.abspath(a["--input"]), a["--output"]
emissive_scale = float(a.get("--emissive-scale", 1.0))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src, use_image_search=True)
scene = bpy.context.scene

cameras = [o for o in scene.objects if o.type == "CAMERA"]
for cam in cameras:
    common.camera_pose(cam)
for cam in cameras:
    bpy.data.objects.remove(cam, do_unlink=True)
if a.get("--skip-lights"):
    for obj in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    common.log("LOST", "lights: the file's lights were left out (--skip-lights)")
for obj in scene.objects:
    if obj.type == "LIGHT":
        common.log("NOTE", f"light {obj.name} {obj.data.type} {obj.data.energy:.4g} W colour {tuple(round(c, 3) for c in obj.data.color)}")

textures_dir = None
for candidate in ("Textures", "textures"):
    if os.path.isdir(os.path.join(os.path.dirname(src), candidate)):
        textures_dir = os.path.join(os.path.dirname(src), candidate)
files = {}
if textures_dir:
    for name in os.listdir(textures_dir):
        files[name.lower()] = os.path.join(textures_dir, name)

loaded = {}


def load(path, non_color=False):
    if path not in loaded:
        img = bpy.data.images.load(path, check_existing=True)
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        loaded[path] = img
    return loaded[path]


def pixels(img):
    buf = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(-1, 4)


flipped = {}


def opengl_normal(path):
    """A copy of a DirectX normal map with green flipped, packed into the .blend so the exporter writes it."""
    if path in flipped:
        return flipped[path]
    src_img = load(path, non_color=True)
    w, h = src_img.size
    px = pixels(src_img)
    px[:, 1] = 1.0 - px[:, 1]
    img = bpy.data.images.new(os.path.splitext(os.path.basename(path))[0] + "_gl", w, h, alpha=False, float_buffer=False, is_data=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(px.ravel())
    img.file_format = "PNG"
    img.pack()
    img.buffers_free()
    del px
    flipped[path] = img
    bpy.data.images.remove(src_img)
    loaded.pop(path, None)
    return img


def has_holes(img):
    if img.channels < 4 or img.depth not in (32, 128):
        return False
    alpha = pixels(img)[:, 3]
    holes = bool((alpha < 0.5).mean() > 0.001)
    del alpha
    img.buffers_free()  # decoded pixels are float RGBA: keeping every texture's would not fit in memory
    return holes


def texture_names(material):
    """The ORCA texture files of a material: from the images the importer attached, else by the material's name."""
    names = set()
    if material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and node.image.filepath:
                names.add(os.path.basename(bpy.path.abspath(node.image.filepath)))
    stems = {re.sub(r"_(BaseColor|Specular|Normal|Emissive)\.dds$", "", n, flags=re.I) for n in names}
    stems.discard("")
    if not stems:
        stems = {f"{material.name}_0"}
    found = {}
    for stem in stems:
        for kind in ("BaseColor", "Specular", "Normal", "Emissive"):
            path = files.get(f"{stem}_{kind}.dds".lower())
            if path:
                found[kind] = path
    return found


def orca_material(material, report):
    tex = texture_names(material)
    if not tex:
        report.setdefault("no ORCA textures found: kept the importer's material", []).append(material.name)
        return
    nt = material.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    i = bsdf.inputs
    if "BaseColor" in tex:
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = load(tex["BaseColor"])
        nt.links.new(node.outputs["Color"], i["Base Color"])
        if has_holes(node.image):
            rnd = nt.nodes.new("ShaderNodeMath")
            rnd.operation = "ROUND"
            nt.links.new(node.outputs["Alpha"], rnd.inputs[0])
            nt.links.new(rnd.outputs["Value"], i["Alpha"])
            report.setdefault("cut-out from the base colour's alpha", []).append(material.name)
    if "Specular" in tex:
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = load(tex["Specular"], non_color=True)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(node.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], i["Roughness"])
        nt.links.new(sep.outputs["Blue"], i["Metallic"])
        report.setdefault("ambient occlusion (Specular.R) not exported", []).append(material.name)
    else:
        i["Roughness"].default_value = 0.6
    if "Normal" in tex:
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = opengl_normal(tex["Normal"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(node.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], i["Normal"])
    if "Emissive" in tex:
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = load(tex["Emissive"])
        nt.links.new(node.outputs["Color"], i["Emission Color"])
        i["Emission Strength"].default_value = emissive_scale
        report.setdefault(f"emissive texture x {emissive_scale:g}", []).append(material.name)


if a.get("--orca"):
    report = {}
    for material in bpy.data.materials:
        if material.users and material.use_nodes:
            orca_material(material, report)
    for what, names in report.items():
        kind = "LOST" if "not" in what or "no ORCA" in what else "NOTE"
        common.log(kind, f"material {what} [{len(names)}: {', '.join(names[:6])}{' ...' if len(names) > 6 else ''}]")
else:
    common.simplify_materials()
common.area_lights_to_spots()
common.log("LOST", "environment: the HDR sky / reflection maps that come with the FBX (glTF has no environment light)")
common.export_glb(out)
