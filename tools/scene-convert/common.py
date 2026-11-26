"""Shared helpers of the scene converters. Runs inside Blender (`blender --background --factory-startup`).

Every converter prints one line per change that loses something, so `convert.log` says what the .glb is missing:

    LOST <kind>: <what>        something of the source the glTF cannot carry (or this converter does not)
    NOTE <kind>: <what>        a choice the converter made (an approximation, a starting value "to tune")
    WROTE <path> ...           the output
    CAMERA position=(x, y, z) look_at=(x, y, z)
                               the source's own camera, in the glTF's frame (Y up, metres): the manifest's start pose
    SUN direction=(x, y, z) illuminance=N
                               the source's sun, not exported: the manifest's `sun: Sun(...)` (see sun_to_manifest)

The glTF material model is one metallic-roughness BSDF per material, plus emission, transmission, IOR, alpha and a
normal map. `simplify_material` rewrites each Cycles node tree into exactly that: one Principled BSDF whose inputs
are either constants or image textures read straight from the source's images (with their UV map and mapping).
Anything in between (procedural textures, colour ramps, mixes of two images, bump from a height map, ...) is
dropped and listed.
"""

import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

# How strong a light a Blender area lamp becomes when it is exported as a spot light (see `area_lights_to_spots`).
AREA_TO_SPOT_POWER = 2.0  # to tune: spreads the lamp's flux over a hemisphere instead of its cosine lobe


def log(kind, text):
    print(f"{kind} {text}", flush=True)


def args():
    """The converter's own arguments: everything after `--` on Blender's command line, as {--name: value}."""
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[argv[i]] = argv[i + 1]
                i += 2
            else:
                out[argv[i]] = True
                i += 1
        else:
            i += 1
    return out


# --- Scene clean-up --------------------------------------------------------------------------------------------


def keep_render_visible():
    """Deletes what the source's render would not show; shows everything it would (so the exporter sees it)."""
    scene = bpy.context.scene
    for layer_collection in walk_layer_collections(bpy.context.view_layer.layer_collection):
        if layer_collection.exclude or layer_collection.collection.hide_render:
            for obj in list(layer_collection.collection.all_objects):
                if obj.name in bpy.data.objects:
                    log("NOTE", f"removed: {obj.name} (its collection is not rendered)")
                    bpy.data.objects.remove(obj, do_unlink=True)
            layer_collection.exclude = False
        layer_collection.hide_viewport = False
        layer_collection.collection.hide_viewport = False
    for obj in list(scene.objects):
        if obj.hide_render:
            log("NOTE", f"removed: {obj.name} (hidden from render)")
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        obj.hide_viewport = False
        try:
            obj.hide_set(False)
        except RuntimeError:
            pass


def walk_layer_collections(layer_collection):
    yield layer_collection
    for child in layer_collection.children:
        yield from walk_layer_collections(child)


def realize_particles():
    """Particle instances (objects / collections scattered by particle systems) become real objects; hair
    strands cannot be exported and are listed."""
    for obj in list(bpy.context.scene.objects):
        for system in obj.particle_systems:
            settings = system.settings
            if settings.type == "HAIR" and settings.render_type == "PATH":
                log("LOST", f"hair: {obj.name}/{system.name} ({len(system.particles)} strands; glTF has no hair)")
            elif settings.render_type in {"OBJECT", "COLLECTION"}:
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.duplicates_make_real()
                log("NOTE", f"particles made real: {obj.name}/{system.name} ({len(system.particles)} instances)")
    for obj in bpy.context.scene.objects:
        if obj.type == "CURVES":
            log("LOST", f"hair curves: {obj.name} (glTF has no hair)")
        if obj.type == "VOLUME":
            log("LOST", f"volume: {obj.name}")


def realize_lit_instances():
    """Collection instances that hold a lamp, or an object whose material is linked to the object rather than its
    mesh, become real objects. Otherwise the lamps inside skip every lamp step (their node strength is not folded
    in: Classroom's ceiling lamps exported at 1 W instead of 20 W) and the exporter takes the mesh's material
    instead of the object's (Classroom's lamp bulbs exported as clear glass instead of their emission)."""
    def needs(collection):
        return any(o.type == "LIGHT" or any(s.link == "OBJECT" for s in o.material_slots)
                   for o in collection.all_objects)

    for obj in list(bpy.context.scene.objects):
        if obj.instance_type != "COLLECTION" or obj.instance_collection is None or not needs(obj.instance_collection):
            continue
        name = obj.instance_collection.name
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)
        log("NOTE", f"instance made real (holds lamps or object materials): {obj.name} ({name})")


def object_materials_to_mesh():
    """A material slot linked to the object (not its mesh) is ignored by Blender's glTF exporter, which writes the
    mesh's own material instead (Classroom's lamp bulbs: clear glass instead of their emission). Such an object gets
    its own copy of the mesh with the object's materials in the mesh's slots."""
    count = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not any(slot.link == "OBJECT" for slot in obj.material_slots):
            continue
        materials = [slot.material for slot in obj.material_slots]
        obj.data = obj.data.copy()
        for index, (slot, material) in enumerate(zip(obj.material_slots, materials)):
            slot.link = "DATA"
            obj.data.materials[index] = material
        count += 1
    if count:
        log("NOTE", f"object-linked materials moved onto a copy of the mesh: {count} objects")


def remove_volume_only():
    """Objects whose every material is a volume (fog boxes) would export as opaque boxes: they are removed."""
    for obj in list(bpy.context.scene.objects):
        mats = [slot.material for slot in obj.material_slots if slot.material]
        if obj.type != "MESH" or not mats:
            continue

        def volume_only(m):
            if not m.use_nodes:
                return False
            outs = [n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL"]
            return bool(outs) and all(not o.inputs["Surface"].is_linked and o.inputs["Volume"].is_linked for o in outs)

        if all(volume_only(m) for m in mats):
            log("LOST", f"volume: {obj.name} (a {mats[0].name} volume box, removed: glTF has no volumes)")
            bpy.data.objects.remove(obj, do_unlink=True)


def note_world():
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return
    for node in world.node_tree.nodes:
        if node.type == "TEX_ENVIRONMENT" and node.image:
            log("LOST", f"world: environment image {node.image.name} (glTF has no environment light)")
        elif node.type in {"TEX_SKY"}:
            log("LOST", f"world: sky texture ({node.sky_type}) (glTF has no environment light)")
        elif node.type == "BACKGROUND" and not node.inputs["Color"].is_linked:
            color = tuple(round(c, 3) for c in node.inputs["Color"].default_value[:3])
            log("LOST", f"world: background colour {color} x {node.inputs['Strength'].default_value:.3g}")


def portal_side(material):
    """'back' or 'front' when `material` is a daylight portal: an emission that shows on one side of a surface and
    a Transparent BSDF on the other, switched by Geometry > Backfacing (Classroom's windows). None otherwise."""
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    output = next((n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    surface = output.inputs["Surface"] if output else None
    if surface is None or not surface.is_linked or surface.links[0].from_node.type != "MIX_SHADER":
        return None
    mix = surface.links[0].from_node
    fac = mix.inputs[0].links[0] if mix.inputs[0].is_linked else None
    if fac is None or fac.from_node.type != "NEW_GEOMETRY" or fac.from_socket.name != "Backfacing":
        return None
    kinds = [s.links[0].from_node.type if s.is_linked else None for s in mix.inputs[1:3]]
    if kinds == ["BSDF_TRANSPARENT", "EMISSION"]:
        return "back"  # Fac 1 (seen from behind) takes the second shader
    if kinds == ["EMISSION", "BSDF_TRANSPARENT"]:
        return "front"
    return None


def light_portals():
    """Cycles daylight portals: a surface over a window that is transparent from outside and emits the sky into the
    room (often invisible to the camera and to shadows). glTF can carry neither the one-sided emission nor 'casts no
    shadow', so exported as they are they become opaque sheets that block the sun on every rung and, on the traced
    rungs, emit out of the room (Solari emits only from a triangle's front face). They are deleted: the sky they
    stood for is the viewer's (its atmosphere, lit by the scene's sun), seen through the opening they covered.
    Blender's own portal lamps (area lights flagged as portals; not lights at all) are deleted for the same reason."""
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT" and obj.data.type == "AREA" and getattr(obj.data.cycles, "is_portal", False):
            log("LOST", f"light portal: {obj.name} (portal lamp; the viewer's sky lights the opening)")
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        if obj.type != "MESH" or not obj.material_slots:
            continue
        sides = {portal_side(slot.material) for slot in obj.material_slots}
        if None in sides:
            continue
        log("LOST", f"light portal: {obj.name} (sky emission on its {'/'.join(sorted(sides))} side; "
                    f"{len(obj.data.polygons)} faces; deleted, the viewer's sky lights the opening)")
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0 and portal_side(material):
            bpy.data.materials.remove(material)


def sun_to_manifest():
    """glTF lights reach Bevy without shadow maps (its loader keeps the lights' defaults), so a sun in the file lights
    a room through its walls on Raster. The viewer's own sun has shadow maps, so a single sun lamp is not exported:
    its direction and illuminance are printed as a SUN line for the manifest's `sun: Sun(...)`, in the units of the
    exporter's lamps (683 lm/W: W/m2 -> lux). Its colour is lost (the viewer's sun is white)."""
    suns = [o for o in bpy.context.scene.objects if o.type == "LIGHT" and o.data.type == "SUN"]
    if len(suns) != 1:
        if suns:
            log("NOTE", f"{len(suns)} sun lamps: exported as they are (no shadow maps on Raster)")
        return
    sun = suns[0]
    d = (sun.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    lux = sun.data.energy * 683.0
    colour = tuple(round(c, 3) for c in sun.data.color)
    log("SUN", f"direction=({d.x:.3f}, {d.z:.3f}, {-d.y:.3f}) illuminance={lux:.0f}")
    log("NOTE", f"sun lamp {sun.name}: {sun.data.energy:.4g} W/m2 -> {lux:.0f} lux, not exported (see SUN)")
    if colour != (1.0, 1.0, 1.0):
        log("LOST", f"sun colour {colour}: {sun.name} (the viewer's sun is white)")
    bpy.data.objects.remove(sun, do_unlink=True)


def camera_pose(obj=None):
    """Prints the scene camera's pose in the glTF frame (Y up): x, z, -y of Blender's."""
    obj = obj or bpy.context.scene.camera
    if obj is None:
        return
    world = obj.matrix_world
    position = world.translation.copy()
    forward = (world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    clip = obj.data.clip_start if obj.type == "CAMERA" else 0.0
    if clip > 0.3:
        # A source camera that looks through a wall with its near clip: start past the clip instead.
        position += forward * clip
        log("NOTE", f"camera {obj.name}: near clip {clip:.2f} m, start pose moved forward past it")
    target = position + forward * 3.0

    def gl(v):
        return f"({v.x:.3f}, {v.z:.3f}, {-v.y:.3f})"

    lens = obj.data.angle if obj.type == "CAMERA" else 0.0
    log("CAMERA", f"position={gl(position)} look_at={gl(target)} fov_deg={math.degrees(lens):.1f} name={obj.name}")


def fold_light_nodes():
    """A lamp's node tree is folded into its power and colour, which is all glTF carries: the Emission strength (a
    constant, or the Light Falloff node feeding it) multiplies its watts, and the Emission colour (a constant, or a
    Blackbody node) multiplies its colour (brightest channel 1, the rest into the watts). Then the tree's nodes are
    deleted, because Blender's glTF exporter reads a lamp's tree even with `use_nodes` off, and through a Light
    Falloff node it ignores the lamp's watts and divides by 4 pi twice (Classroom's ceiling lamps came out at 1/12.6
    of their power). Each light datablock is folded once, however many lamps share it. The falloff curve is lost."""
    done = set()
    for obj in bpy.context.scene.objects:
        if obj.type != "LIGHT" or obj.data.node_tree is None or obj.data.name in done:
            continue
        light = obj.data
        done.add(light.name)
        tree = light.node_tree
        output = next((n for n in tree.nodes if n.type == "OUTPUT_LIGHT" and n.is_active_output), None)
        linked = output is not None and output.inputs["Surface"].is_linked
        emission = output.inputs["Surface"].links[0].from_node if linked else None
        if emission is not None and emission.type == "EMISSION":
            sock = emission.inputs["Strength"]
            strength = sock.default_value
            if sock.is_linked:
                src = sock.links[0].from_node
                if src.type == "LIGHT_FALLOFF":
                    strength = src.inputs["Strength"].default_value
                    log("LOST", f"light falloff: {obj.name} ({sock.links[0].from_socket.name}; exported as inverse-square)")
                else:
                    log("LOST", f"light strength from {src.bl_idname}: {obj.name} (its default {strength:.4g} kept)")
            colour_in = emission.inputs["Color"]
            colour = tuple(colour_in.default_value)[:3]
            if colour_in.is_linked:
                src = colour_in.links[0].from_node
                if src.type == "BLACKBODY" and not src.inputs["Temperature"].is_linked:
                    colour = blackbody(src.inputs["Temperature"].default_value)[:3]
                    log("NOTE", f"light colour from Blackbody {src.inputs['Temperature'].default_value:.0f} K: {obj.name}")
                else:
                    colour = (1.0, 1.0, 1.0)
                    log("LOST", f"light colour from {src.bl_idname}: {obj.name} (white kept)")
            peak = max(colour) if max(colour) > 0 else 1.0
            light.color = tuple(c * e / peak for c, e in zip(light.color, colour))
            factor = strength * peak
            if abs(factor - 1.0) > 1e-6:
                log("NOTE", f"light node strength folded in: {obj.name} {light.energy:.4g} W x {factor:.4g}")
                light.energy *= factor
        elif linked:
            log("LOST", f"light nodes: {obj.name} (surface from {emission.bl_idname}; only its colour and power are exported)")
        extra = sorted({n.type for n in tree.nodes} - {"EMISSION", "OUTPUT_LIGHT", "LIGHT_FALLOFF", "BLACKBODY"})
        if extra:
            log("LOST", f"light nodes: {obj.name} ({', '.join(extra)}; only its colour and power are exported)")
        tree.nodes.clear()


def area_lights_to_spots():
    """glTF has no area light. Each area lamp becomes a spot light at its centre, aimed where it shines, with a
    180-degree cone and full blend (a hemisphere), power x AREA_TO_SPOT_POWER. Its shape and softness are lost.
    First every lamp's node tree is folded into its power and colour (fold_light_nodes)."""
    fold_light_nodes()
    for obj in list(bpy.context.scene.objects):
        if obj.type != "LIGHT" or obj.data.type != "AREA":
            continue
        area = obj.data
        spot = bpy.data.lights.new(name=f"{area.name}_spot", type="SPOT")
        spot.color = area.color
        spot.energy = area.energy * AREA_TO_SPOT_POWER
        spot.spot_size = math.pi - 0.02  # Bevy needs the outer angle below 90 degrees
        spot.spot_blend = 1.0
        spot.shadow_soft_size = max(area.size, getattr(area, "size_y", area.size)) * 0.5
        size = f"{area.size:.2f} x {getattr(area, 'size_y', area.size):.2f} m"
        obj.data = spot
        log("NOTE", f"area light -> spot: {obj.name} ({size}, {area.energy:.4g} W -> {spot.energy:.4g} W, hemisphere cone)")


# --- Materials ---------------------------------------------------------------------------------------------------

PASS_THROUGH = {
    "MIX_RGB": "Color1",
    "MIX": None,
    "HUE_SAT": "Color",
    "BRIGHTCONTRAST": "Color",
    "GAMMA": "Color",
    "INVERT": "Color",
    "CURVE_RGB": "Color",
    "VALTORGB": "Fac",
    "MATH": None,
    "MAP_RANGE": None,
    "SEPRGB": "Image",
    "SEPARATE_COLOR": "Color",
    "SEPXYZ": "Vector",
    "RGBTOBW": "Color",
    "NORMAL_MAP": "Color",
    "BUMP": "Height",
    "REROUTE": None,
    "CLAMP": None,
    "DISPLACEMENT": "Height",
}


class Ctx:
    """Where a socket lives: the stack of group nodes we are inside (to step out through Group Input)."""

    def __init__(self, stack=()):
        self.stack = tuple(stack)


def from_socket(socket, ctx, seen=None):
    """The node and output socket feeding `socket` (following reroutes and group boundaries), or (None, None)."""
    if not socket.is_linked:
        return None, None, ctx
    link = socket.links[0]
    if not link.is_valid or link.is_muted:
        return None, None, ctx
    node, out = link.from_node, link.from_socket
    if node.type == "REROUTE":
        return from_socket(node.inputs[0], ctx)
    if node.type == "GROUP_INPUT":
        if not ctx.stack:
            return None, None, ctx
        group = ctx.stack[-1]
        outer = Ctx(ctx.stack[:-1])
        inner = group.inputs.get(out.name) or group.inputs[list(node.outputs).index(out)]
        if not inner.is_linked:
            return ("CONST", inner), None, outer
        return from_socket(inner, outer)
    if node.type == "GROUP" and node.node_tree:
        # Step into the group: its output node's input of the same name.
        outputs = [n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output] or [
            n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT"
        ]
        if outputs:
            inner_ctx = Ctx(ctx.stack + (node,))
            target = outputs[0].inputs.get(out.name)
            if target is not None:
                if not target.is_linked:
                    return ("CONST", target), None, inner_ctx
                return from_socket(target, inner_ctx)
    return node, out, ctx


def const(socket):
    value = socket.default_value
    try:
        return tuple(value)
    except TypeError:
        return value


def find_image(socket, ctx, lost, what, depth=0):
    """The image texture node upstream of `socket` (through simple colour nodes, which are dropped), or None."""
    node, out, ctx = from_socket(socket, ctx)
    if node is None or isinstance(node, tuple):
        return None, ctx
    if node.type == "TEX_IMAGE":
        if node.image is None:
            lost.append(f"{what}: image node without image")
            return None, ctx
        return node, ctx
    if depth > 8:
        return None, ctx
    if node.type in PASS_THROUGH:
        if node.type not in {"REROUTE", "SEPRGB", "SEPARATE_COLOR", "NORMAL_MAP"}:
            lost.append(f"{what}: {node.bl_idname} between image and shader dropped")
        name = PASS_THROUGH[node.type]
        candidates = [node.inputs[name]] if name and name in node.inputs else list(node.inputs)
        for candidate in candidates:
            if candidate.is_linked and candidate.type in {"RGBA", "VALUE", "VECTOR"}:
                found, found_ctx = find_image(candidate, ctx, lost, what, depth + 1)
                if found:
                    return found, found_ctx
        return None, ctx
    if node.type.startswith("TEX_"):
        lost.append(f"{what}: procedural {node.bl_idname}")
    elif node.type == "VERTEX_COLOR" or node.type == "ATTRIBUTE":
        lost.append(f"{what}: vertex colour / attribute input")
    else:
        lost.append(f"{what}: input from {node.bl_idname}")
    return None, ctx


def blackbody(kelvin):
    """The colour of Blender's Blackbody node: Planck's law through the CIE 1931 observer (the analytic fit of Wyman,
    Sloan and Shirley 2013) into linear Rec.709, divided by its luminance as Cycles does (5800 K: 1.11, 0.98, 0.92)."""
    def g(l, mu, s1, s2):
        return math.exp(-0.5 * ((l - mu) / (s1 if l < mu else s2)) ** 2)

    x = y = z = 0.0
    for nm in range(380, 781, 5):
        m = nm * 1e-9
        planck = 1.0 / (m ** 5 * (math.exp(1.4387769e-2 / (m * max(kelvin, 800.0))) - 1.0))
        x += planck * (1.056 * g(nm, 599.8, 37.9, 31.0) + 0.362 * g(nm, 442.0, 16.0, 26.7) - 0.065 * g(nm, 501.1, 20.4, 26.2))
        y += planck * (0.821 * g(nm, 568.8, 46.9, 40.5) + 0.286 * g(nm, 530.9, 16.3, 31.1))
        z += planck * (1.217 * g(nm, 437.0, 11.8, 36.0) + 0.681 * g(nm, 459.0, 26.0, 13.8))
    rgb = (3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z)
    rgb = [max(c, 0.0) for c in rgb]
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return tuple(c / lum for c in rgb) + (1.0,)


def value_or_image(socket, ctx, lost, what):
    """(constant, image node) for a shader input."""
    node, _, sub = from_socket(socket, ctx)
    if isinstance(node, tuple):
        return const(node[1]), None
    if node is None:
        return const(socket), None
    if node.type == "BLACKBODY" and not node.inputs["Temperature"].is_linked:
        return blackbody(node.inputs["Temperature"].default_value), None
    image, image_ctx = find_image(socket, ctx, lost, what)
    return const(socket), image


class Surface:
    """One glTF material's worth of inputs."""

    def __init__(self):
        self.base = (0.8, 0.8, 0.8, 1.0)
        self.base_image = None
        self.roughness = 0.5
        self.roughness_image = None
        self.metallic = 0.0
        self.metallic_image = None
        self.normal_image = None
        self.normal_strength = 1.0
        self.emission = (0.0, 0.0, 0.0, 1.0)
        self.emission_image = None
        self.emission_strength = 0.0
        self.transmission = 0.0
        self.ior = 1.5
        self.alpha = 1.0
        self.alpha_image = None
        self.alpha_from_base = False


def principled(node, ctx, lost):
    s = Surface()
    inputs = node.inputs
    s.base, s.base_image = value_or_image(inputs["Base Color"], ctx, lost, "base colour")
    s.roughness, s.roughness_image = value_or_image(inputs["Roughness"], ctx, lost, "roughness")
    s.metallic, s.metallic_image = value_or_image(inputs["Metallic"], ctx, lost, "metallic")
    s.alpha, s.alpha_image = value_or_image(inputs["Alpha"], ctx, lost, "alpha")
    s.ior = const(inputs["IOR"]) if "IOR" in inputs else 1.5
    trans = inputs.get("Transmission Weight") or inputs.get("Transmission")
    if trans is not None:
        s.transmission = const(trans) if not trans.is_linked else 1.0
    emission = inputs.get("Emission Color") or inputs.get("Emission")
    s.emission, s.emission_image = value_or_image(emission, ctx, lost, "emission")
    strength = inputs.get("Emission Strength")
    s.emission_strength = const(strength) if strength is not None else 1.0
    normal_node, _, nctx = from_socket(inputs["Normal"], ctx)
    if normal_node is not None and not isinstance(normal_node, tuple):
        if normal_node.type == "NORMAL_MAP":
            s.normal_strength = const(normal_node.inputs["Strength"])
            s.normal_image, _ = find_image(normal_node.inputs["Color"], nctx, lost, "normal")
        elif normal_node.type == "BUMP":
            lost.append("normal: bump from a height map (glTF has no bump)")
            under, _, uctx = from_socket(normal_node.inputs["Normal"], nctx)
            if under is not None and not isinstance(under, tuple) and under.type == "NORMAL_MAP":
                s.normal_image, _ = find_image(under.inputs["Color"], uctx, lost, "normal")
        else:
            lost.append(f"normal: from {normal_node.bl_idname}")
    for name in ("Subsurface Weight", "Coat Weight", "Sheen Weight"):
        sock = inputs.get(name)
        if sock is not None and (sock.is_linked or (const(sock) or 0) > 0.01):
            lost.append(f"{name.split()[0].lower()} (not exported by this converter)")
    return s


def evaluate(socket, ctx, lost, depth=0):
    """The Surface a shader socket amounts to."""
    node, out, ctx = from_socket(socket, ctx)
    if node is None or isinstance(node, tuple) or depth > 12:
        lost.append("surface: no shader")
        return Surface()
    t = node.type
    if t == "BSDF_PRINCIPLED":
        return principled(node, ctx, lost)
    s = Surface()
    if t in {"BSDF_DIFFUSE", "BSDF_OREN_NAYAR"}:
        s.base, s.base_image = value_or_image(node.inputs["Color"], ctx, lost, "base colour")
        s.roughness = 0.9
    elif t in {"BSDF_GLOSSY", "BSDF_ANISOTROPIC"}:
        s.base, s.base_image = value_or_image(node.inputs["Color"], ctx, lost, "base colour")
        s.roughness, s.roughness_image = value_or_image(node.inputs["Roughness"], ctx, lost, "roughness")
        s.metallic = 1.0
    elif t in {"BSDF_GLASS", "BSDF_REFRACTION"}:
        s.base, s.base_image = value_or_image(node.inputs["Color"], ctx, lost, "base colour")
        s.roughness = const(node.inputs["Roughness"])
        s.ior = const(node.inputs["IOR"])
        s.transmission = 1.0
    elif t == "BSDF_TRANSLUCENT":
        s.base, s.base_image = value_or_image(node.inputs["Color"], ctx, lost, "base colour")
        lost.append("translucency")
    elif t == "EMISSION":
        s.base = (0.0, 0.0, 0.0, 1.0)
        s.emission, s.emission_image = value_or_image(node.inputs["Color"], ctx, lost, "emission")
        s.emission_strength = const(node.inputs["Strength"])
    elif t == "BSDF_TRANSPARENT":
        s.alpha = 0.0
    elif t == "HOLDOUT":
        s.alpha = 0.0
        lost.append("holdout")
    elif t == "SUBSURFACE_SCATTERING":
        s.base, s.base_image = value_or_image(node.inputs["Color"], ctx, lost, "base colour")
        lost.append("subsurface scattering")
    elif t == "ADD_SHADER":
        a = evaluate(node.inputs[0], ctx, lost, depth + 1)
        b = evaluate(node.inputs[1], ctx, lost, depth + 1)
        main, other = (a, b) if a.emission_strength == 0 or b.emission_strength > 0 else (b, a)
        if other.emission_strength > 0 or other.emission_image:
            main.emission, main.emission_image = other.emission, other.emission_image
            main.emission_strength = other.emission_strength
        lost.append("add shader: kept one surface and the emission")
        return main
    elif t == "MIX_SHADER":
        fac = node.inputs[0]
        a = evaluate(node.inputs[1], ctx, lost, depth + 1)
        b = evaluate(node.inputs[2], ctx, lost, depth + 1)
        a_clear = a.alpha == 0.0 and a.alpha_image is None
        b_clear = b.alpha == 0.0 and b.alpha_image is None
        fac_node, _, fctx = from_socket(fac, ctx)
        if a_clear or b_clear:
            # Transparent mixed with a surface: the factor is the alpha (an image's alpha, or a constant).
            surface = b if a_clear else a
            if fac_node is None or isinstance(fac_node, tuple):
                f = const(fac) if fac_node is None else const(fac_node[1])
                surface.alpha = f if a_clear else 1.0 - f
            else:
                image, _ = find_image(fac, ctx, lost, "alpha")
                surface.alpha_image = image
                if not a_clear:
                    lost.append("alpha: inverted factor approximated as opaque cutout")
            return surface
        if fac_node is None or isinstance(fac_node, tuple):
            f = const(fac) if fac_node is None else const(fac_node[1])
            picked = b if f >= 0.5 else a
            lost.append(f"mix shader: kept the {'second' if f >= 0.5 else 'first'} shader (factor {f:.2f})")
            return picked
        # A factor from Fresnel / Layer Weight over diffuse + glossy is a dielectric: diffuse colour, glossy roughness.
        if a.metallic == 0.0 and b.metallic == 1.0 and not b.base_image:
            a.roughness, a.roughness_image = b.roughness, b.roughness_image
            lost.append("mix shader: diffuse + glossy read as one dielectric")
            return a
        if b.metallic == 0.0 and a.metallic == 1.0 and not a.base_image:
            b.roughness, b.roughness_image = a.roughness, a.roughness_image
            lost.append("mix shader: glossy + diffuse read as one dielectric")
            return b
        lost.append(f"mix shader with a varying factor ({fac_node.bl_idname}): kept the first shader")
        return a
    elif t == "GROUP":
        return Surface()
    else:
        lost.append(f"surface: {node.bl_idname} read as grey")
    return s


_uv_names = set()  # the UV maps every mesh of the material being simplified has


def copy_image_node(nodes, links, source, location):
    """A new image node reading the same image the same way (UV map and mapping kept when they are plain)."""
    image = nodes.new("ShaderNodeTexImage")
    image.image = source.image
    image.interpolation = source.interpolation
    image.extension = source.extension
    image.location = location
    vec = source.inputs["Vector"]
    if vec.is_linked:
        upstream = vec.links[0].from_node
        mapping_node = None
        uv_node = None
        if upstream.type == "MAPPING":
            mapping_node = upstream
            if upstream.inputs["Vector"].is_linked:
                uv_node = upstream.inputs["Vector"].links[0].from_node
        else:
            uv_node = upstream
        target = image.inputs["Vector"]
        if mapping_node is not None:
            mapping = nodes.new("ShaderNodeMapping")
            mapping.vector_type = mapping_node.vector_type
            for name in ("Location", "Rotation", "Scale"):
                if name in mapping_node.inputs:
                    mapping.inputs[name].default_value = mapping_node.inputs[name].default_value
            mapping.location = (location[0] - 200, location[1])
            links.new(mapping.outputs["Vector"], target)
            target = mapping.inputs["Vector"]
        if uv_node is not None:
            if uv_node.type == "UVMAP" and uv_node.uv_map and uv_node.uv_map not in _uv_names:
                # A UV map some mesh of this material lacks: the exporter would write texCoord -1.
                return image, f"image {source.image.name} on UV map {uv_node.uv_map!r} some meshes lack, read with the first UVs"
            if uv_node.type == "UVMAP" and uv_node.uv_map:
                uv = nodes.new("ShaderNodeUVMap")
                uv.uv_map = uv_node.uv_map
                uv.location = (location[0] - 400, location[1])
                links.new(uv.outputs["UV"], target)
            elif uv_node.type == "TEX_COORD" and vec.links[0].from_socket.name != "UV":
                return image, f"image {source.image.name} projected with {vec.links[0].from_socket.name} coordinates, read with UVs"
            elif uv_node.type not in {"TEX_COORD", "UVMAP"}:
                return image, f"image {source.image.name} coordinates from {uv_node.bl_idname}, read with UVs"
    if source.projection != "FLAT":
        return image, f"image {source.image.name} {source.projection} projection, read with UVs"
    return image, None


def simplify_material(material):
    """Rewrites a node material as one Principled BSDF the glTF exporter maps one-to-one. Returns what was lost."""
    if not material.use_nodes or material.node_tree is None:
        return []
    tree = material.node_tree
    outputs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"]
    output = next((n for n in outputs if n.is_active_output and n.target in {"ALL", "CYCLES"}), None) or (
        outputs[0] if outputs else None
    )
    if output is None:
        return ["no material output"]
    surface_in = output.inputs["Surface"]
    # Already exportable as it is: a Principled BSDF straight into the output, images straight into it.
    lost = []
    s = evaluate(surface_in, Ctx(), lost)
    if output.inputs["Volume"].is_linked:
        lost.append("volume shader")
    if output.inputs["Displacement"].is_linked:
        lost.append("displacement")

    nodes, links = tree.nodes, tree.links
    keep_images = {id(n) for n in (s.base_image, s.roughness_image, s.metallic_image, s.normal_image,
                                   s.emission_image, s.alpha_image) if n is not None}
    # Build the new tree beside the old one, then delete the old nodes.
    old = list(nodes)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def colour(v):
        if isinstance(v, (int, float)):
            return (v, v, v, 1.0)
        v = tuple(v)
        return v + (1.0,) * (4 - len(v)) if len(v) < 4 else v[:4]

    def scalar(v):
        if isinstance(v, (int, float)):
            return float(v)
        v = tuple(v)
        return sum(v[:3]) / 3.0

    y = 300

    def image_into(source, socket, channel=None, non_color=False):
        nonlocal y
        node, problem = copy_image_node(nodes, links, source, (-400, y))
        y -= 300
        if problem:
            lost.append(problem)
        if non_color and node.image.colorspace_settings.name not in {"Non-Color", "Raw", "Linear"}:
            node.image.colorspace_settings.name = "Non-Color"
        if channel is None:
            links.new(node.outputs["Color"], socket)
        elif channel == "A":
            links.new(node.outputs["Alpha"], socket)
        else:
            sep = nodes.new("ShaderNodeSeparateColor")
            sep.location = (-100, y + 300)
            links.new(node.outputs["Color"], sep.inputs["Color"])
            links.new(sep.outputs[{"R": "Red", "G": "Green", "B": "Blue"}[channel]], socket)
        return node

    inputs = bsdf.inputs
    inputs["Base Color"].default_value = colour(s.base)
    base_node = image_into(s.base_image, inputs["Base Color"]) if s.base_image else None
    inputs["Roughness"].default_value = min(max(scalar(s.roughness), 0.0), 1.0)
    if s.roughness_image:
        image_into(s.roughness_image, inputs["Roughness"], "G" if s.roughness_image == s.metallic_image else None, True)
    inputs["Metallic"].default_value = min(max(scalar(s.metallic), 0.0), 1.0)
    if s.metallic_image:
        image_into(s.metallic_image, inputs["Metallic"], "B" if s.roughness_image == s.metallic_image else None, True)
    if s.normal_image:
        nm = nodes.new("ShaderNodeNormalMap")
        nm.location = (0, -600)
        nm.inputs["Strength"].default_value = scalar(s.normal_strength)
        image_into(s.normal_image, nm.inputs["Color"], None, True)
        links.new(nm.outputs["Normal"], inputs["Normal"])
    strength = scalar(s.emission_strength)
    emission = colour(s.emission)
    if s.emission_image or (strength > 0 and max(emission[:3]) > 0):
        inputs["Emission Color"].default_value = emission
        inputs["Emission Strength"].default_value = strength
        if s.emission_image:
            image_into(s.emission_image, inputs["Emission Color"])
    inputs["IOR"].default_value = scalar(s.ior) if s.ior else 1.5
    if s.transmission > 0:
        inputs["Transmission Weight"].default_value = min(scalar(s.transmission), 1.0)
    alpha_socket = inputs["Alpha"]
    if s.alpha_image:
        # Cut-out: the exporter writes alphaMode MASK (cutoff 0.5) for alpha through a Math "Round".
        rnd = nodes.new("ShaderNodeMath")
        rnd.operation = "ROUND"
        rnd.location = (0, -300)
        if base_node is not None and s.alpha_image.image == base_node.image:
            links.new(base_node.outputs["Alpha"], rnd.inputs[0])
        else:
            image_into(s.alpha_image, rnd.inputs[0], "A" if s.alpha_image.image.depth == 32 else None, True)
        links.new(rnd.outputs["Value"], alpha_socket)
    else:
        alpha_socket.default_value = scalar(s.alpha)
    for node in old:
        nodes.remove(node)
    return lost


def simplify_materials():
    """Every material, simplified; one LOST line per distinct loss with the materials it hit."""
    seen = {}
    for material in bpy.data.materials:
        if material.users == 0:
            continue
        global _uv_names
        meshes = [o.data for o in bpy.context.scene.objects if o.type == "MESH"
                  and any(slot.material == material for slot in o.material_slots)]
        _uv_names = set.intersection(*[{uv.name for uv in m.uv_layers} for m in meshes]) if meshes else set()
        try:
            lost = simplify_material(material)
        except Exception as error:  # a node layout this converter does not know: keep the material as it was
            lost = [f"converter error ({error.__class__.__name__}: {error}); material left as it was"]
        for item in dict.fromkeys(lost):
            seen.setdefault(item, []).append(material.name)
    for item, materials in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        names = ", ".join(materials[:6]) + (f" and {len(materials) - 6} more" if len(materials) > 6 else "")
        log("LOST", f"material {item} [{len(materials)}: {names}]")


def photometric_emission():
    """Blender's exporter writes emission strength as it is, but converts lamp watts to candela at 683 lm/W. Bevy
    reads emissive as nits (cd/m2) and lamps as lumens, so a physically lit source (Cycles, Mitsuba) needs its
    emission in the same units as its lamps: radiance = strength / pi W/(m2 sr), times 683 lm/W."""
    scale = 683.0 / math.pi
    count = 0
    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED" and node.inputs["Emission Strength"].default_value > 0:
                colour = node.inputs["Emission Color"]
                if colour.is_linked or max(colour.default_value[:3]) > 0:
                    node.inputs["Emission Strength"].default_value *= scale
                    count += 1
    log("NOTE", f"emission x {scale:.1f} (W/m2 exitance -> nits, the lamps' 683 lm/W) on {count} materials")


# --- Export ------------------------------------------------------------------------------------------------------


def stats():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    tris = 0
    for obj in meshes:
        tris += sum(len(p.vertices) - 2 for p in obj.data.polygons)
    lights = [o for o in bpy.context.scene.objects if o.type == "LIGHT"]
    kinds = {}
    for light in lights:
        kinds[light.data.type] = kinds.get(light.data.type, 0) + 1
    log("NOTE", f"scene: {len(meshes)} mesh objects, ~{tris} triangles before modifiers, lights {kinds}")


def export_glb(path, **extra):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stats()
    final = path
    # Written under a temporary name and renamed when complete, so a killed run leaves no half file behind.
    path = os.path.join(os.path.dirname(os.path.abspath(final)), ".partial-" + os.path.basename(final))
    options = dict(
        filepath=path,
        export_format="GLB",
        use_active_scene=True,
        export_apply=True,
        export_lights=True,
        export_cameras=False,  # the viewer brings its own camera; the source's pose is printed as CAMERA
        export_extras=False,
        export_yup=True,
        export_image_format="AUTO",
        export_materials="EXPORT",
        export_animations=False,
    )
    options.update(extra)
    bpy.ops.export_scene.gltf(**options)
    check_glb(path)
    os.replace(path, final)
    path = final
    log("WROTE", f"{path} {os.path.getsize(path) / 1e6:.1f} MB")


def check_glb(path):
    """The exporter writes texCoord -1 for a texture on a UV map a mesh lacks (seen in Barbershop), which glTF
    loaders reject. Such texture references are removed (the material keeps its factors) and listed; any other
    negative index fails the conversion."""
    import json
    import struct

    with open(path, "rb") as f:
        data = f.read()
    length, _ = struct.unpack("<II", data[12:20])
    doc = json.loads(data[20 : 20 + length])
    rest = data[20 + length :]
    dropped, bad = [], []
    for material in doc.get("materials", []):
        for holder in (material, material.get("pbrMetallicRoughness", {})):
            for key in [k for k, v in holder.items() if isinstance(v, dict) and v.get("texCoord", 0) < 0]:
                del holder[key]
                dropped.append(f"{material.get('name')}.{key}")

    def walk(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, int) and item < 0 and key in {"index", "texCoord", "source", "sampler", "material", "mesh", "light"}:
                    bad.append(f"{where}.{key}={item}")
                walk(item, f"{where}.{key}")
        elif isinstance(value, list):
            for n, item in enumerate(value):
                walk(item, f"{where}[{n}]")

    walk(doc, "$")
    if bad:
        raise SystemExit(f"invalid glTF written: {bad[:5]}")
    if dropped:
        log("LOST", f"textures on a UV map the mesh lacks (exporter wrote texCoord -1), removed [{len(dropped)}: {', '.join(dropped[:6])}{' ...' if len(dropped) > 6 else ''}]")
        text = json.dumps(doc, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        total = 12 + 8 + len(text) + len(rest)
        with open(path, "wb") as f:
            f.write(data[:8] + struct.pack("<I", total) + struct.pack("<II", len(text), 0x4E4F534A) + text + rest)


# --- Bounded conversion (tools/scene-convert/bounded.py) ---------------------------------------------------------


def defer_image_loads():
    """Importers (FBX's above all) create and decode every texture they meet: Emerald Square's 331 DDS files cost
    4.9 GB of decoded pixels the converter then throws away. While this is on, an importer's
    `bpy_extras.image_utils.load_image` makes a 1x1 placeholder that remembers the file in ["source_path"];
    `load_deferred_images(load)` later swaps each placeholder still in use for `load(path)`."""
    from bpy_extras import image_utils

    def placeholder(filepath, dirname="", *rest, **kwargs):
        path = filepath if os.path.isabs(filepath) or not dirname else os.path.join(dirname, filepath)
        if not os.path.exists(path) and kwargs.get("recursive") and dirname:
            for root, _, names in os.walk(dirname):  # what use_image_search does
                if os.path.basename(filepath) in names:
                    path = os.path.join(root, os.path.basename(filepath))
                    break
        img = bpy.data.images.new(os.path.basename(filepath), 1, 1)
        img["source_path"] = os.path.normpath(path)
        return img

    image_utils.load_image = placeholder


def source_path(img):
    """The file an image comes from, whether a deferred placeholder or a loaded image."""
    return img.get("source_path") or bpy.path.abspath(img.filepath)


def load_deferred_images(load):
    swapped = 0
    for img in [i for i in bpy.data.images if "source_path" in i]:
        if img.users and os.path.exists(img["source_path"]):
            img.user_remap(load(img["source_path"]))
            swapped += 1
        bpy.data.images.remove(img)
    if swapped:
        log("NOTE", f"textures loaded from the importer's placeholders: {swapped}")


def finish(out):
    """The converter's last step: the .glb at `out`, or with --stage-dir DIR, the scene saved for bounded.py
    (DIR/scene.blend + DIR/objects.json) so it is exported in chunks, each by its own Blender process."""
    stage = args().get("--stage-dir")
    if not stage:
        export_glb(out)
        return
    import json

    stats()
    scene = bpy.context.scene
    # Flat: each object keeps its world transform and loses its parent, and empties go, so a chunk never needs
    # an object of another chunk.
    for obj in scene.objects:
        if obj.parent is not None:
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
    for obj in [o for o in scene.objects if o.type not in {"MESH", "LIGHT"}]:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)
    objects = []
    for obj in scene.objects:
        tris = sum(len(p.vertices) - 2 for p in obj.data.polygons) if obj.type == "MESH" else 0
        objects.append({"name": obj.name, "type": obj.type, "data": obj.data.name, "tris": tris})
    os.makedirs(stage, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(stage, ".partial-scene.blend"), compress=False)
    with open(os.path.join(stage, "objects.json"), "w") as f:
        json.dump(objects, f)
    os.replace(os.path.join(stage, ".partial-scene.blend"), os.path.join(stage, "scene.blend"))
    log("STAGED", f"{stage}: {len(objects)} objects")
