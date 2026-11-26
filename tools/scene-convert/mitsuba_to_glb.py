"""A Mitsuba 0.5/0.6 scene (Benedikt Bitterli's rendering resources) to .glb for gfx_viewer.

    blender --background --factory-startup --python tools/scene-convert/mitsuba_to_glb.py -- \
        --input scene_v0.6.xml --output Scene.glb

Geometry: every <shape type="obj"> is imported with Blender's OBJ importer (axes untouched) and placed with its
toWorld matrix; rectangle, cube, sphere and disk shapes are built. The file's own Y-up frame is kept: the glTF has
the same coordinates as the XML. Meshes used by several shapes are imported once and shared.

Materials: each BSDF becomes one glTF metallic-roughness material (approximate, "to tune"):
    diffuse                    base colour = reflectance, roughness 1
    plastic / roughplastic     base colour = diffuseReflectance, roughness = sqrt(alpha) (0.05 when smooth), IOR
    conductor / roughconductor metallic 1, base colour = normal-incidence Fresnel F0 from eta and k (or the named
                               metal; material "none" = a perfect mirror), times specularReflectance
    dielectric / rough / thin  transmission 1, white, IOR = intIOR, roughness = sqrt(alpha)
    twosided                   the inner BSDF, double-sided (one-sided BSDFs export single-sided)
    mask                       the inner BSDF with the opacity texture as a cut-out (alpha MASK 0.5)
    bumpmap                    the inner BSDF; the bump map is LOST (glTF has no bump)
    coating / roughcoating     the inner BSDF; the coat is LOST
Textures: "bitmap" textures keep their file and uscale/vscale; other procedural textures are LOST.
Media inside shapes (<medium>, sigmaA/sigmaS) are LOST.

Lights: an area emitter on a shape keeps its surface emissive (radiance -> emissive colour x strength) AND gets a
punctual light for rasterised lighting: a hemisphere spot at a rectangle's / disk's centre facing its normal, a
point light at a sphere's centre, power = the emitter's flux (pi x area x radiance) x 2 for the spot (its flux
spread evenly over the hemisphere) and x 1 for the point. On traced rungs, which also light from emissive
surfaces, those emitters therefore count twice. Environment, sun and sky emitters are LOST.
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# Normal-incidence reflectance of Mitsuba's named conductors (linear RGB, approximate). to tune
METALS = {
    "Ag": (0.97, 0.96, 0.91), "Al": (0.91, 0.92, 0.92), "Au": (1.0, 0.78, 0.34), "Cu": (0.96, 0.64, 0.54),
    "Cr": (0.55, 0.55, 0.55), "Fe": (0.56, 0.57, 0.58), "Ni": (0.66, 0.61, 0.53), "Ti": (0.54, 0.50, 0.45),
    "W": (0.51, 0.50, 0.47), "Hg": (0.78, 0.78, 0.78), "none": (1.0, 1.0, 1.0),
}
SMOOTH_ROUGHNESS = 0.05  # to tune: what a "smooth" Mitsuba BSDF becomes (0 makes Raster's reflections alias)

a = common.args()
src, out = os.path.abspath(a["--input"]), a["--output"]
base_dir = os.path.dirname(src)
tree = ET.parse(src)
root_xml = tree.getroot()
ids = {el.get("id"): el for el in root_xml.iter() if el.get("id") and el.tag != "ref"}

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
root = bpy.data.objects.new("mitsuba_root", None)
scene.collection.objects.link(root)
# The XML is Y-up; Blender is Z-up. Everything is built in the XML's frame under this root, which the exporter's
# own Z-up -> Y-up conversion undoes: the glTF ends up in the XML's coordinates.
root.matrix_world = Matrix.Rotation(math.radians(90.0), 4, "X")

lost = {}


def lose(what, where):
    lost.setdefault(what, set()).add(where)


def floats(text):
    return [float(v) for v in text.replace(",", " ").split()]


def rgb(el):
    """An <rgb>, <spectrum> or <float> element as an RGB triple."""
    if el.tag in {"rgb", "srgb"}:
        v = floats(el.get("value"))
        if len(v) == 1:
            v = v * 3
        if el.tag == "srgb":
            v = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in v]
        return tuple(v[:3])
    if el.tag == "spectrum":
        v = el.get("value")
        if ":" in v:
            lose("spectrum read as its average", el.get("name"))
            vals = [float(p.split(":")[1]) for p in v.replace(",", " ").split()]
            m = sum(vals) / len(vals)
            return (m, m, m)
        v = floats(v)
        return (v[0], v[0], v[0])
    if el.tag == "float":
        v = float(el.get("value"))
        return (v, v, v)
    return None


def param(el, name):
    for child in el:
        if child.get("name") == name:
            return child
    return None


def transform(el):
    """The toWorld matrix of a shape or sensor."""
    t = param(el, "toWorld")
    m = Matrix.Identity(4)
    if t is None:
        return m
    for op in t:
        if op.tag == "matrix":
            v = floats(op.get("value"))
            m = Matrix([v[0:4], v[4:8], v[8:12], v[12:16]]) @ m
        elif op.tag == "translate":
            m = Matrix.Translation((float(op.get("x", 0)), float(op.get("y", 0)), float(op.get("z", 0)))) @ m
        elif op.tag == "scale":
            if op.get("value"):
                s = float(op.get("value"))
                m = Matrix.Diagonal((s, s, s, 1.0)) @ m
            else:
                m = Matrix.Diagonal((float(op.get("x", 1)), float(op.get("y", 1)), float(op.get("z", 1)), 1.0)) @ m
        elif op.tag == "rotate":
            axis = Vector((float(op.get("x", 0)), float(op.get("y", 0)), float(op.get("z", 0))))
            m = Matrix.Rotation(math.radians(float(op.get("angle"))), 4, axis) @ m
        elif op.tag == "lookat":
            origin = Vector(floats(op.get("origin")))
            target = Vector(floats(op.get("target")))
            up = Vector(floats(op.get("up", "0 1 0")))
            z = (target - origin).normalized()
            x = up.cross(z).normalized()
            y = z.cross(x)
            look = Matrix((x.to_4d(), y.to_4d(), z.to_4d(), origin.to_4d())).transposed()
            look[3][3] = 1.0
            for i in range(3):
                look[3][i] = 0.0
            m = look @ m
    return m


# --- Materials ---------------------------------------------------------------------------------------------------

images = {}


def image(path):
    full = os.path.join(base_dir, path)
    if full not in images:
        images[full] = bpy.data.images.load(full, check_existing=True)
    return images[full]


class Spec:
    def __init__(self):
        self.base = (0.8, 0.8, 0.8)
        self.base_tex = None
        self.uv_scale = (1.0, 1.0)
        self.roughness = 1.0
        self.metallic = 0.0
        self.transmission = 0.0
        self.ior = 1.5
        self.alpha_tex = None
        self.alpha = 1.0
        self.two_sided = False


def texture_or_rgb(el, name, spec, where):
    p = param(el, name)
    if p is None:
        return None, None
    if p.tag == "ref":
        p = ids[p.get("id")]
    if p.tag == "texture":
        if p.get("type") == "bitmap":
            filename = param(p, "filename").get("value")
            us, vs = param(p, "uscale"), param(p, "vscale")
            if us is not None or vs is not None:
                spec.uv_scale = (float(us.get("value")) if us is not None else 1.0,
                                 float(vs.get("value")) if vs is not None else 1.0)
            return None, filename
        lose(f"procedural texture {p.get('type')} read as grey", where)
        return (0.5, 0.5, 0.5), None
    return rgb(p), None


def alpha_roughness(el):
    p = param(el, "alpha")
    if p is None:
        return 0.1 ** 0.5  # Mitsuba's default alpha 0.1
    if p.tag == "texture":
        lose("roughness texture read as constant", "")
        return 0.3
    return math.sqrt(float(p.get("value")))


def f0(el):
    material = param(el, "material")
    if material is not None:
        return METALS.get(material.get("value"), (0.9, 0.9, 0.9))
    eta, k = param(el, "eta"), param(el, "k")
    if eta is None or k is None:
        return METALS["Cu"]  # Mitsuba's default conductor is copper
    n, kk = rgb(eta), rgb(k)
    return tuple(((n[i] - 1) ** 2 + kk[i] ** 2) / ((n[i] + 1) ** 2 + kk[i] ** 2) for i in range(3))


def read_bsdf(el, spec, where):
    if el.tag == "ref":
        el = ids[el.get("id")]
    kind = el.get("type")
    if kind == "twosided":
        spec.two_sided = True
        read_bsdf(next(c for c in el if c.tag in {"bsdf", "ref"}), spec, where)
    elif kind == "mask":
        v, tex = texture_or_rgb(el, "opacity", spec, where)
        if tex:
            spec.alpha_tex = tex
        elif v:
            spec.alpha = sum(v) / 3.0
        read_bsdf(next(c for c in el if c.tag in {"bsdf", "ref"}), spec, where)
    elif kind in {"bumpmap", "normalmap"}:
        lose(f"{kind} (glTF has no bump; normalmap not read)", where)
        read_bsdf(next(c for c in el if c.tag in {"bsdf", "ref"}), spec, where)
    elif kind in {"coating", "roughcoating"}:
        lose("clear coat", where)
        read_bsdf(next(c for c in el if c.tag in {"bsdf", "ref"}), spec, where)
    elif kind == "diffuse" or kind == "roughdiffuse":
        v, tex = texture_or_rgb(el, "reflectance", spec, where)
        spec.base, spec.base_tex = v or (0.5, 0.5, 0.5), tex
        spec.roughness = 1.0
    elif kind in {"plastic", "roughplastic"}:
        v, tex = texture_or_rgb(el, "diffuseReflectance", spec, where)
        spec.base, spec.base_tex = v or (0.5, 0.5, 0.5), tex
        spec.roughness = alpha_roughness(el) if kind == "roughplastic" else SMOOTH_ROUGHNESS
        ior = param(el, "intIOR")
        spec.ior = float(ior.get("value")) if ior is not None and ior.tag == "float" else 1.49
    elif kind in {"conductor", "roughconductor"}:
        f = f0(el)
        s = param(el, "specularReflectance")
        if s is not None:
            sv = rgb(s) if s.tag != "texture" else (1, 1, 1)
            f = tuple(f[i] * sv[i] for i in range(3))
        spec.base, spec.metallic = f, 1.0
        spec.roughness = alpha_roughness(el) if kind == "roughconductor" else SMOOTH_ROUGHNESS * 0.4
    elif kind in {"dielectric", "roughdielectric", "thindielectric"}:
        spec.base, spec.transmission = (1.0, 1.0, 1.0), 1.0
        ior = param(el, "intIOR")
        spec.ior = float(ior.get("value")) if ior is not None and ior.tag == "float" else 1.5046
        spec.roughness = alpha_roughness(el) if kind == "roughdielectric" else 0.0
        if kind == "thindielectric":
            spec.two_sided = True
    elif kind == "null":
        spec.alpha = 0.0
    else:
        lose(f"bsdf {kind} read as grey diffuse", where)


materials = {}


def material_for(bsdf_el, name, emission=None):
    key = (id(bsdf_el), emission)
    if key in materials:
        return materials[key]
    spec = Spec()
    if bsdf_el is not None:
        read_bsdf(bsdf_el, spec, name)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    i = bsdf.inputs
    i["Base Color"].default_value = (*spec.base, 1.0)
    i["Roughness"].default_value = min(max(spec.roughness, 0.0), 1.0)
    i["Metallic"].default_value = spec.metallic
    i["IOR"].default_value = spec.ior
    i["Transmission Weight"].default_value = spec.transmission
    mat.use_backface_culling = not spec.two_sided
    uv = None
    if spec.uv_scale != (1.0, 1.0):
        uv = nt.nodes.new("ShaderNodeMapping")
        uv.inputs["Scale"].default_value = (spec.uv_scale[0], spec.uv_scale[1], 1.0)
        coords = nt.nodes.new("ShaderNodeUVMap")
        nt.links.new(coords.outputs["UV"], uv.inputs["Vector"])
    base_node = None
    if spec.base_tex:
        base_node = nt.nodes.new("ShaderNodeTexImage")
        base_node.image = image(spec.base_tex)
        nt.links.new(base_node.outputs["Color"], i["Base Color"])
        if uv:
            nt.links.new(uv.outputs["Vector"], base_node.inputs["Vector"])
    if spec.alpha_tex:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image(spec.alpha_tex)
        tex.image.colorspace_settings.name = "Non-Color"
        if uv:
            nt.links.new(uv.outputs["Vector"], tex.inputs["Vector"])
        rnd = nt.nodes.new("ShaderNodeMath")
        rnd.operation = "ROUND"
        nt.links.new(tex.outputs["Color"], rnd.inputs[0])
        nt.links.new(rnd.outputs["Value"], i["Alpha"])
    elif spec.alpha < 1.0:
        i["Alpha"].default_value = spec.alpha
    if emission is not None:
        strength = max(emission)
        if strength > 0:
            i["Emission Color"].default_value = (*(c / strength for c in emission), 1.0)
            i["Emission Strength"].default_value = strength * math.pi  # Blender's strength is exitance
    materials[key] = mat
    return mat


# --- Shapes ------------------------------------------------------------------------------------------------------

meshes = {}
lights = 0


def obj_mesh(path):
    if path in meshes:
        return meshes[path]
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=os.path.join(base_dir, path), forward_axis="Y", up_axis="Z",
                          use_split_objects=False, use_split_groups=False)
    new = [o for o in bpy.data.objects if o not in before]
    mesh = None
    for obj in new:
        if obj.type == "MESH" and mesh is None:
            mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None:
        mesh.materials.clear()
    meshes[path] = mesh
    return mesh


def prim_mesh(kind, el):
    if kind == "rectangle":
        bpy.ops.mesh.primitive_plane_add(size=2.0)
    elif kind == "disk":
        bpy.ops.mesh.primitive_circle_add(radius=1.0, vertices=48, fill_type="TRIFAN")
    elif kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=48, ring_count=24)
        bpy.ops.object.shade_smooth()
    obj = bpy.context.active_object
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


def add_light(kind, matrix, power, colour, name, size):
    global lights
    data = bpy.data.lights.new(name, kind)
    data.energy = power
    data.color = colour
    data.shadow_soft_size = size
    if kind == "SPOT":
        data.spot_size = math.pi - 0.02
        data.spot_blend = 1.0
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.parent = root
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = matrix
    lights += 1


shape_count = 0
for shape in root_xml.iter("shape"):
    kind = shape.get("type")
    name = f"{kind}_{shape_count:04d}"
    shape_count += 1
    matrix = transform(shape)
    if kind == "obj":
        mesh = obj_mesh(param(shape, "filename").get("value"))
        name = os.path.splitext(os.path.basename(param(shape, "filename").get("value")))[0]
    elif kind in {"rectangle", "disk", "cube", "sphere"}:
        mesh = prim_mesh(kind, shape)
        if kind == "sphere":
            radius = float(param(shape, "radius").get("value")) if param(shape, "radius") is not None else 1.0
            c = param(shape, "center")
            centre = Vector((float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 0)))) if c is not None else Vector()
            matrix = matrix @ Matrix.Translation(centre) @ Matrix.Diagonal((radius, radius, radius, 1.0))
    else:
        lose(f"shape type {kind}", name)
        continue
    if mesh is None:
        lose("obj file without a mesh", name)
        continue
    if any(c.tag == "medium" or (c.tag == "ref" and c.get("name") in {"interior", "exterior"}) for c in shape):
        lose("participating medium inside a shape", name)
    bsdf_el = next((c for c in shape if c.tag == "bsdf" or (c.tag == "ref" and ids.get(c.get("id"), ET.Element("x")).tag == "bsdf")), None)
    if bsdf_el is not None and bsdf_el.tag == "ref":
        bsdf_el = ids[bsdf_el.get("id")]  # shared BSDFs become one shared material
    emitter = next((c for c in shape if c.tag == "emitter"), None)
    radiance = None
    if emitter is not None and emitter.get("type") == "area":
        radiance = rgb(param(emitter, "radiance"))
    elif emitter is not None:
        lose(f"emitter {emitter.get('type')} on a shape", name)
    mat_name = bsdf_el.get("id") if bsdf_el is not None and bsdf_el.get("id") else name
    if radiance is not None:
        mat_name = f"{mat_name}_emitter"
    mat = material_for(bsdf_el, mat_name, radiance)
    obj = bpy.data.objects.new(name, mesh)
    obj.data = mesh
    scene.collection.objects.link(obj)
    if len(obj.data.materials) == 0:
        obj.data.materials.append(None)
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = mat
    obj.parent = root
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = matrix
    if radiance is not None:
        # Flux of the emitter, from its area in world space (sum of transformed triangle areas).
        area = 0.0
        m3 = matrix.to_3x3()
        verts = [matrix @ v.co for v in mesh.vertices]
        for poly in mesh.polygons:
            pts = [verts[i] for i in poly.vertices]
            for j in range(1, len(pts) - 1):
                area += (pts[j] - pts[0]).cross(pts[j + 1] - pts[0]).length * 0.5
        flux = math.pi * area * max(radiance)
        colour = tuple(c / max(radiance) for c in radiance)
        centre = sum(verts, Vector()) / max(len(verts), 1)
        if kind in {"rectangle", "disk"}:
            normal = (m3 @ Vector((0, 0, 1))).normalized()
            # A light shines along its local -Z.
            rot = normal.to_track_quat("-Z", "Y").to_matrix().to_4x4()
            add_light("SPOT", Matrix.Translation(centre) @ rot, flux * 2.0, colour, f"{name}_light", math.sqrt(area) * 0.5)
            common.log("NOTE", f"area emitter -> emissive {kind} + hemisphere spot: {name} ({area:.2f} m2, radiance {radiance}, spot {flux * 2:.4g} W)")
        else:
            size = (area / (4 * math.pi)) ** 0.5
            add_light("POINT", Matrix.Translation(centre), flux, colour, f"{name}_light", size)
            common.log("NOTE", f"area emitter -> emissive {kind} + point light: {name} ({area:.2f} m2, radiance {radiance}, point {flux:.4g} W)")

for emitter in root_xml.findall("emitter"):
    lose(f"scene emitter {emitter.get('type')} (environment / sun / sky)", "scene")

sensor = root_xml.find("sensor")
if sensor is not None:
    m = transform(sensor)
    pos = m @ Vector((0, 0, 0))
    look = m @ Vector((0, 0, 1))
    direction = (look - pos).normalized()
    target = pos + direction * 3.0
    fov = param(sensor, "fov")
    common.log("CAMERA", f"position=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) look_at=({target.x:.3f}, {target.y:.3f}, {target.z:.3f}) fov_deg={fov.get('value') if fov is not None else '?'}")

for what, where in sorted(lost.items()):
    places = sorted(w for w in where if w)
    common.log("LOST", f"{what} [{len(places)}: {', '.join(places[:6])}{' ...' if len(places) > 6 else ''}]")
common.log("NOTE", f"{shape_count} shapes, {len(meshes)} distinct OBJ meshes, {len(materials)} materials, {lights} lights")
common.photometric_emission()
common.export_glb(out)
