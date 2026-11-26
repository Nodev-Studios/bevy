# Converts an NVIDIA ORCA scene shipped as FBX + DDS textures (Amazon Lumberyard Bistro, Beeple's Zero Day) into
# one .glb that Bevy loads, for gfx_viewer. Run it through Blender, headless only (tools/fetch-scenes.sh does):
#
#   blender --background --factory-startup --python tools/orca-fbx-to-gltf.py -- \
#       --fbx IN.fbx --out OUT.glb [--max-texture 1024] [--emissive-strength 1.0]
#
# ORCA's texture convention (each scene's README.txt): <Material>_BaseColor (RGB colour, A opacity),
# <Material>_Specular (R occlusion, G roughness, B metalness), <Material>_Normal (DirectX, two channels in BC5),
# <Material>_Emissive (RGB). Blender's FBX importer wires the Specular texture to "Specular IOR Level" and leaves
# the emissive texture out, so every material is rebuilt here from its textures:
#
#   - base colour; its alpha becomes a cutout (glTF MASK at 0.5) when the texture has cut-out texels, else the
#     material is opaque (the importer marks every material as blended, which Bevy would sort and draw forward);
#   - roughness and metalness from the Specular texture's G and B (glTF's metallicRoughness layout);
#   - the normal map, turned from DirectX to OpenGL (green flipped) with its blue channel rebuilt;
#   - emission from <Material>_Emissive when it exists, times --emissive-strength (KHR_materials_emissive_strength);
#   - glass-like materials (name has Glass, Wine, Water, Beer, Ice, Liquid) become transmissive.
#
# What is lost (gfx_viewer's report lists it): the occlusion channel (glTF wants it as a separate texture slot the
# exporter does not take from this layout), the FBX lights and camera path (Blender's importer brings no light
# objects from these files), the pyscene extras (wine volume absorption, index of refraction), texture resolution
# above --max-texture, and the DDS mipmaps (PNG/JPEG carry none; Bevy makes none for them).
import argparse
import json
import os
import shutil
import struct
import sys
import time

import bpy
import numpy as np

GLASS = ("glass", "wine", "water", "beer", "ice", "liquid")


def parse():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--fbx")
    p.add_argument("--fix-only", action="store_true", help="only repair --out (see fix_glb)")
    p.add_argument("--out", required=True)
    p.add_argument("--max-texture", type=int, default=1024)  # to tune: VRAM with no mipmaps
    p.add_argument("--emissive-strength", type=float, default=1.0)
    return p.parse_args(argv)


def principled(mat):
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def texture_paths(mat):
    """The ORCA textures a material uses, by kind, from the image nodes the importer made."""
    found = {}
    for node in mat.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        path = bpy.path.abspath(node.image.filepath)
        stem = os.path.splitext(os.path.basename(path))[0]
        for kind in ("BaseColor", "Specular", "Normal", "Emissive"):
            if stem.endswith("_" + kind):
                found.setdefault(kind, path)
    if "BaseColor" in found and "Emissive" not in found:
        base = found["BaseColor"]
        stem, ext = os.path.splitext(base)
        for candidate in (stem[: -len("BaseColor")] + "Emissive" + ext,):
            if os.path.isfile(candidate):
                found["Emissive"] = candidate
    return found


class Textures:
    def __init__(self, out_dir, max_size):
        self.out_dir = out_dir
        self.max_size = max_size
        self.done = {}
        os.makedirs(out_dir, exist_ok=True)

    def get(self, path, kind):
        """The converted image for `path` (PNG or JPEG next to the output), and whether it has cut-out texels."""
        key = (path, kind)
        if key in self.done:
            return self.done[key]
        source = bpy.data.images.load(path, check_existing=False)
        if kind in ("Specular", "Normal"):
            source.colorspace_settings.name = "Non-Color"
        w, h = source.size
        if w == 0 or h == 0:
            self.done[key] = (None, False)
            return self.done[key]
        scale = min(1.0, self.max_size / max(w, h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        if (nw, nh) != (w, h):
            source.scale(nw, nh)
        px = np.empty(nw * nh * 4, dtype=np.float32)
        source.pixels.foreach_get(px)
        px = px.reshape(-1, 4)
        cutout = False
        if kind == "Normal":
            x = px[:, 0] * 2.0 - 1.0
            y = -(px[:, 1] * 2.0 - 1.0)  # DirectX -> OpenGL
            z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
            px[:, 0] = x * 0.5 + 0.5
            px[:, 1] = y * 0.5 + 0.5
            px[:, 2] = z * 0.5 + 0.5
            px[:, 3] = 1.0
        elif kind == "BaseColor":
            cutout = bool((px[:, 3] < 0.5).mean() > 0.001)
            if not cutout:
                px[:, 3] = 1.0
        else:
            px[:, 3] = 1.0
        source.pixels.foreach_set(px.ravel())
        stem = os.path.splitext(os.path.basename(path))[0]
        lossless = kind == "Normal" or cutout
        ext = ".png" if lossless else ".jpg"
        target = os.path.join(self.out_dir, stem + ext)
        source.filepath_raw = target
        source.file_format = "PNG" if lossless else "JPEG"
        if lossless:
            source.save()
        else:
            source.save(filepath=target, quality=90)
        # The image now is the saved file (the exporter writes its bytes as they are).
        source.name = stem
        self.done[key] = (source, cutout)
        return self.done[key]


def rebuild(mat, textures, emissive_strength, counts):
    if mat.node_tree is None:
        return
    bsdf = principled(mat)
    if bsdf is None:
        return
    paths = texture_paths(mat)
    nt = mat.node_tree
    for node in list(nt.nodes):
        if node.type in ("TEX_IMAGE", "NORMAL_MAP", "SEPARATE_COLOR", "MATH"):
            nt.nodes.remove(node)
    for link in list(nt.links):
        if link.to_node == bsdf:
            nt.links.remove(link)
    glass = any(word in mat.name.lower() for word in GLASS)
    cutout = False

    def image_node(image, y):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = image
        node.location = (-700, y)
        return node

    if "BaseColor" in paths:
        image, cutout = textures.get(paths["BaseColor"], "BaseColor")
        if image is not None:
            node = image_node(image, 300)
            nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
            if cutout and not glass:
                cut = nt.nodes.new("ShaderNodeMath")
                cut.operation = "GREATER_THAN"
                cut.inputs[1].default_value = 0.5
                cut.location = (-350, 150)
                nt.links.new(node.outputs["Alpha"], cut.inputs[0])
                nt.links.new(cut.outputs[0], bsdf.inputs["Alpha"])
                counts["cutout"] += 1
    if "Specular" in paths:
        image, _ = textures.get(paths["Specular"], "Specular")
        if image is not None:
            node = image_node(image, 0)
            split = nt.nodes.new("ShaderNodeSeparateColor")
            split.location = (-400, 0)
            nt.links.new(node.outputs["Color"], split.inputs["Color"])
            nt.links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
            nt.links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])
    if "Normal" in paths:
        image, _ = textures.get(paths["Normal"], "Normal")
        if image is not None:
            node = image_node(image, -300)
            normal = nt.nodes.new("ShaderNodeNormalMap")
            normal.location = (-350, -300)
            nt.links.new(node.outputs["Color"], normal.inputs["Color"])
            nt.links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
    if "Emissive" in paths:
        image, _ = textures.get(paths["Emissive"], "Emissive")
        if image is not None:
            node = image_node(image, -600)
            nt.links.new(node.outputs["Color"], bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = emissive_strength
            counts["emissive"] += 1
    if glass:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["Metallic"].default_value = 0.0
        counts["glass"] += 1
    mat.blend_method = "CLIP" if cutout and not glass else "OPAQUE"
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    counts["materials"] += 1


def fix_glb(path):
    """Repairs what Blender's exporter writes and Bevy refuses: a texture reference with texCoord -1 (a texture
    whose mesh lacks the UV map it names) loses the index, so it reads UV 0."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    length, kind = struct.unpack_from("<I4s", data, 12)
    assert magic == b"glTF" and kind == b"JSON"
    gltf = json.loads(data[20 : 20 + length])
    fixed = 0

    def walk(value):
        nonlocal fixed
        if isinstance(value, dict):
            if value.get("texCoord", 0) < 0:
                del value["texCoord"]
                fixed += 1
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(gltf)
    if fixed == 0:
        return 0
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    rest = data[20 + length :]
    out = struct.pack("<4sII", b"glTF", version, 12 + 8 + len(text) + len(rest))
    out += struct.pack("<I4s", len(text), b"JSON") + text + rest
    with open(path, "wb") as f:
        f.write(out)
    return fixed


def main():
    args = parse()
    if args.fix_only:
        print(f"orca-fbx-to-gltf: {fix_glb(args.out)} texture references repaired in {args.out}")
        return
    started = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=args.fbx)
    print(f"orca-fbx-to-gltf: imported {args.fbx} in {time.time() - started:.0f} s")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    stem = os.path.splitext(os.path.basename(args.out))[0]
    textures = Textures(os.path.join(out_dir, stem + "-textures"), args.max_texture)
    counts = {"materials": 0, "cutout": 0, "emissive": 0, "glass": 0}
    for mat in list(bpy.data.materials):
        rebuild(mat, textures, args.emissive_strength, counts)
    lights = sum(1 for o in bpy.data.objects if o.type == "LIGHT")
    print(
        f"orca-fbx-to-gltf: {counts['materials']} materials rebuilt ({counts['cutout']} cut-out, "
        f"{counts['emissive']} emissive, {counts['glass']} glass), {len(textures.done)} textures, "
        f"{lights} lights; {time.time() - started:.0f} s"
    )
    bpy.ops.export_scene.gltf(
        filepath=args.out,
        export_format="GLB",
        export_image_format="AUTO",
        export_tangents=True,
        export_cameras=False,
        export_lights=True,
        export_yup=True,
        export_animations=False,
    )
    repaired = fix_glb(args.out)
    if repaired:
        print(f"orca-fbx-to-gltf: {repaired} texture references with texCoord -1 now read UV 0")
    # The GLB carries the textures; the converted files were only its input.
    shutil.rmtree(textures.out_dir, ignore_errors=True)
    print(f"orca-fbx-to-gltf: wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - started:.0f} s")


main()
