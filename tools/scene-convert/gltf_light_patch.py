"""Writes a copy of a .gltf with its lights fixed, beside the original (so its relative file paths still resolve).
Plain Python, no Blender:

    python3 -I tools/scene-convert/gltf_light_patch.py --input Scene.gltf --output Scene_day.gltf \
        [--sun-lux 20000] [--drop-zero-lights]

Intel's Sponza ships its sun and its 23 lamps with intensity 0 (they are set in its 3ds Max file, not in the glTF),
which renders black, and the viewer would not add its own sun because the file has lights. --sun-lux sets every
directional light's illuminance; --drop-zero-lights removes the lights left at 0 (they would light nothing and
still cost a shadow map each).
"""

import json
import os
import sys


def main():
    argv = sys.argv[1:]
    opt = {argv[i]: (argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True)
           for i in range(len(argv)) if argv[i].startswith("--")}
    src, out = opt["--input"], opt["--output"]
    if os.path.dirname(os.path.abspath(src)) != os.path.dirname(os.path.abspath(out)):
        raise SystemExit("the output must sit beside the input, or its textures will not be found")
    doc = json.load(open(src))
    ext = doc.get("extensions", {}).get("KHR_lights_punctual", {})
    lights = ext.get("lights", [])
    if "--sun-lux" in opt:
        for light in lights:
            if light.get("type") == "directional":
                print(f"NOTE sun {light.get('name')}: {light.get('intensity', 1)} -> {opt['--sun-lux']} lux (to tune)")
                light["intensity"] = float(opt["--sun-lux"])
    if opt.get("--drop-zero-lights"):
        keep = [i for i, light in enumerate(lights) if light.get("intensity", 1.0) > 0]
        remap = {old: new for new, old in enumerate(keep)}
        dropped = [lights[i].get("name") for i in range(len(lights)) if i not in remap]
        ext["lights"] = [lights[i] for i in keep]
        for node in doc.get("nodes", []):
            ref = node.get("extensions", {}).get("KHR_lights_punctual")
            if ref is not None:
                if ref["light"] in remap:
                    ref["light"] = remap[ref["light"]]
                else:
                    del node["extensions"]["KHR_lights_punctual"]
                    if not node["extensions"]:
                        del node["extensions"]
        print(f"LOST lights at intensity 0 removed [{len(dropped)}: {', '.join(map(str, dropped[:6]))}{' ...' if len(dropped) > 6 else ''}]")
    json.dump(doc, open(out, "w"))
    print(f"WROTE {out}")


main()
