#!/usr/bin/env python3
"""Converts a scene with bounded memory: several Blender processes, each under a memory cap, resumable.

    python3 -I tools/scene-convert/bounded.py --script fbx_to_glb.py --output DIR/Scene.gltf \
        [--memory 8G] [--chunk-tris 400000] [--blender PATH] [--log FILE] [--keep-work] [--orm-occlusion] \
        -- <the script's own arguments, without --output>

Why: one Blender process that imports, rebuilds and exports a whole city block in one go (Emerald Square:
10 M triangles, 331 textures) was killed by the out-of-memory killer at 6.6 GB, twice. Here:

1. stage   the converter (`--script`, any of this folder's that ends in `common.finish`) imports the source and
           fixes its materials, then saves it as WORK/stage/scene.blend instead of exporting (`--stage-dir`).
2. chunks  the scene's objects are packed into chunks of about --chunk-tris unique triangles (every object that
           shares a mesh goes in the same chunk, so instancing is kept); each chunk is exported by its own Blender
           process (export_chunk.py: it appends only that chunk's objects) to a .gltf + .bin. Textures are never
           decoded or re-encoded: the glTF points at the image files the converter wrote.
3. merge   plain Python (no Blender) joins the chunks into one OUT.gltf; each chunk's .bin becomes one buffer in
           OUT-buffers/ (glTF allows several), shared images, samplers, textures and materials are written once.

Every Blender step runs in its own `systemd-run --user --scope` with MemoryMax=--memory and MemorySwapMax=0, so a
step that needs more is killed alone, by its own cap, and never takes the desktop or other jobs with it. Before a
step starts, the machine must have --memory available (else it stops with exit 75 and starts nothing). Each step's
peak (the scope's cgroup memory.peak) is printed as `PEAK <step> <MB>`.

Fails soft: a step killed by the cap prints `KILLED` with what to change, keeps every finished step in WORK
(= OUT's folder/.work-<stem>), and exits 3; rerunning the same command resumes from the first unfinished step.
Lowering --chunk-tris re-plans the chunks (finished chunks of the old plan are not reused); the stage step cannot be
split. WORK is deleted after a successful merge unless --keep-work.

--orm-occlusion: the source's metallic-roughness textures hold ambient occlusion in red (glTF's ORM layout, as
NVIDIA ORCA's _Specular files do): each material also gets that texture as its occlusionTexture.

Exit codes: 0 done, 1 a step failed (not memory), 2 usage, 3 a step went over the memory cap, 75 not enough free
memory to start a step.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEMORY = "8G"  # to tune: per step; Emerald Square's stage step measured ~3.5 GB, its chunks less
DEFAULT_CHUNK_TRIS = 400_000  # to tune: unique triangles per chunk; smaller = lower export peak, more processes


def size_bytes(text):
    units = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    text = text.strip().upper()
    return int(float(text[:-1]) * units[text[-1]]) if text[-1] in units else int(text)


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


# --- one capped step -------------------------------------------------------------------------------------------


def measure(result_path, cmd):
    """Runs inside the capped scope: runs `cmd`, then records its exit and the scope's peak memory and OOM kills.
    The kernel's OOM killer picks the largest process of the scope (Blender), so this small parent survives to
    write the result."""
    code = subprocess.call(cmd)
    cgroup = ""
    with open("/proc/self/cgroup") as f:
        for line in f:
            if line.startswith("0::"):
                cgroup = "/sys/fs/cgroup" + line[3:].strip()
    peak, oom = 0, 0
    try:
        with open(os.path.join(cgroup, "memory.peak")) as f:
            peak = int(f.read().split()[0])
        with open(os.path.join(cgroup, "memory.events")) as f:
            oom = int(dict(line.split() for line in f).get("oom_kill", 0))
    except OSError:
        pass
    with open(result_path, "w") as f:
        json.dump({"returncode": code, "peak": peak, "oom_kill": oom}, f)
    return code


class Killed(Exception):
    pass


class Failed(Exception):
    pass


class Runner:
    def __init__(self, opts, work):
        self.opts, self.work = opts, work
        self.peaks = {}
        self.n = 0

    def blender(self, step, script, args):
        memory = self.opts["memory"]
        need = size_bytes(memory)
        avail = mem_available()
        if avail < need:
            raise MemoryError(
                f"step {step} needs up to {memory} and the machine has only {avail / 2**30:.1f} GB available now; "
                f"nothing was started. Wait for other jobs to finish, or lower --memory (the step may then be killed "
                f"by its cap, which is safe)."
            )
        self.n += 1
        result = os.path.join(self.work, f".result-{os.getpid()}-{self.n}.json")
        unit = f"scene-convert-{os.getpid()}-{self.n}"
        cmd = [
            "systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
            "-p", f"MemoryMax={memory}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue", "--",
            sys.executable, "-I", os.path.abspath(__file__), "_measure", result, "--",
            self.opts["blender"], "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", os.path.join(HERE, script), "--", *args,
        ]
        start = time.time()
        with open(self.opts["log"], "a") as log:
            log.write(f"\n==== {step}: {' '.join(cmd[cmd.index('--background') - 1:])}\n")
            log.flush()
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        # No result: the scope itself was stopped before its small parent could write one (by the cap, if
        # OOMPolicy=continue was not honoured, or from outside); its exit then says SIGKILL or SIGTERM.
        res = {"returncode": code, "peak": 0, "oom_kill": int(code in (-9, -15, 137, 143))}
        if os.path.exists(result):
            with open(result) as f:
                res = json.load(f)
            os.remove(result)
        peak_mb = res["peak"] / 2**20
        self.peaks[step] = round(peak_mb)
        line = f"PEAK {step} {peak_mb:.0f} MB" if res["peak"] else f"PEAK {step} unknown"
        line += f" ({time.time() - start:.0f} s, cap {memory})"
        print(line, flush=True)
        with open(self.opts["log"], "a") as log:
            log.write(line + "\n")
        if res["oom_kill"] or res["returncode"] in (-9, 137):
            raise Killed(step)
        if res["returncode"] != 0 or code != 0:
            raise Failed(f"{step} exited with {res['returncode']}; see {self.opts['log']}")


# --- plan --------------------------------------------------------------------------------------------------------


def plan_chunks(objects, chunk_tris):
    """Objects grouped by the mesh they share, groups packed in order into chunks of about chunk_tris unique
    triangles (a group bigger than that is a chunk of its own). Lights go with the first chunk."""
    groups, order = {}, []
    lights = []
    for obj in objects:
        if obj["type"] == "LIGHT":
            lights.append(obj["name"])
            continue
        if obj["data"] not in groups:
            groups[obj["data"]] = {"tris": obj["tris"], "objects": []}
            order.append(obj["data"])
        groups[obj["data"]]["objects"].append(obj["name"])
    chunks, current, tris = [], [], 0
    for key in order:
        g = groups[key]
        if current and tris + g["tris"] > chunk_tris:
            chunks.append(current)
            current, tris = [], 0
        current += g["objects"]
        tris += g["tris"]
    if current or not chunks:
        chunks.append(current)
    chunks[0] = lights + chunks[0]
    return chunks


# --- merge -------------------------------------------------------------------------------------------------------


def merge(chunk_dirs, out, orm_occlusion):
    """Joins the chunks' .gltf files into `out`; each chunk's .bin becomes one buffer in <stem>-buffers/."""
    out_dir = os.path.dirname(os.path.abspath(out))
    stem = os.path.splitext(os.path.basename(out))[0]
    buf_dir = os.path.join(out_dir, f"{stem}-buffers")
    tmp_buf = os.path.join(out_dir, f".partial-{stem}-buffers")
    shutil.rmtree(tmp_buf, ignore_errors=True)
    os.makedirs(tmp_buf)
    doc = {"asset": {"version": "2.0", "generator": f"tools/scene-convert/bounded.py ({len(chunk_dirs)} chunks)"},
           "scene": 0, "scenes": [{"name": stem, "nodes": []}], "nodes": [], "meshes": [], "materials": [],
           "textures": [], "images": [], "samplers": [], "accessors": [], "bufferViews": [], "buffers": []}
    lights = []
    used, required = set(), set()
    seen = {"images": {}, "samplers": {}, "textures": {}, "materials": {}, "lights": {}}

    def once(kind, item, target):
        key = json.dumps(item, sort_keys=True)
        if key not in seen[kind]:
            seen[kind][key] = len(target)
            target.append(item)
        return seen[kind][key]

    def texture_refs(value, tex_map):
        """Every textureInfo of a material (a dict with "index") points at the merged textures."""
        if isinstance(value, dict):
            if "index" in value and isinstance(value["index"], int):
                value["index"] = tex_map[value["index"]]
            for item in value.values():
                texture_refs(item, tex_map)
        elif isinstance(value, list):
            for item in value:
                texture_refs(item, tex_map)

    dropped = []
    for k, cdir in enumerate(chunk_dirs):
        with open(os.path.join(cdir, "chunk.gltf")) as f:
            c = json.load(f)
        used |= set(c.get("extensionsUsed", []))
        required |= set(c.get("extensionsRequired", []))
        buf_map = {}
        for i, buf in enumerate(c.get("buffers", [])):
            name = f"{k:02}" + (f"-{i}" if i else "") + ".bin"
            os.link(os.path.join(cdir, urllib.parse.unquote(buf["uri"])), os.path.join(tmp_buf, name))
            buf_map[i] = len(doc["buffers"])
            doc["buffers"].append({"byteLength": buf["byteLength"], "uri": f"{stem}-buffers/{name}"})
        bv0, acc0, mesh0, node0 = (len(doc[x]) for x in ("bufferViews", "accessors", "meshes", "nodes"))
        for bv in c.get("bufferViews", []):
            bv["buffer"] = buf_map[bv["buffer"]]
            doc["bufferViews"].append(bv)
        for acc in c.get("accessors", []):
            if "bufferView" in acc:
                acc["bufferView"] += bv0
            for part in ("indices", "values"):
                if part in acc.get("sparse", {}):
                    acc["sparse"][part]["bufferView"] += bv0
            doc["accessors"].append(acc)
        img_map = {}
        for i, img in enumerate(c.get("images", [])):
            if "uri" in img:
                path = os.path.normpath(os.path.join(cdir, urllib.parse.unquote(img["uri"])))
                img = dict(img, uri=urllib.parse.quote(os.path.relpath(path, out_dir)))
            else:
                img["bufferView"] += bv0
            img_map[i] = once("images", img, doc["images"])
        smp_map = {i: once("samplers", s, doc["samplers"]) for i, s in enumerate(c.get("samplers", []))}
        tex_map = {}
        for i, tex in enumerate(c.get("textures", [])):
            if "sampler" in tex:
                tex["sampler"] = smp_map[tex["sampler"]]
            if "source" in tex:
                tex["source"] = img_map[tex["source"]]
            for ext in tex.get("extensions", {}).values():
                if "source" in ext:
                    ext["source"] = img_map[ext["source"]]
            tex_map[i] = once("textures", tex, doc["textures"])
        mat_map = {}
        for i, mat in enumerate(c.get("materials", [])):
            texture_refs(mat, tex_map)
            for holder in (mat, mat.get("pbrMetallicRoughness", {})):
                # Blender writes texCoord -1 for a texture on a UV map the mesh lacks; loaders reject it.
                for key in [x for x, v in holder.items() if isinstance(v, dict) and v.get("texCoord", 0) < 0]:
                    del holder[key]
                    dropped.append(f"{mat.get('name')}.{key}")
            mr = mat.get("pbrMetallicRoughness", {}).get("metallicRoughnessTexture")
            if orm_occlusion and mr and "occlusionTexture" not in mat:
                mat["occlusionTexture"] = {"index": mr["index"], **({"texCoord": mr["texCoord"]} if "texCoord" in mr else {})}
            mat_map[i] = once("materials", mat, doc["materials"])
        light_map = {}
        for i, light in enumerate(c.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])):
            light_map[i] = once("lights", light, lights)
        for mesh in c.get("meshes", []):
            for prim in mesh["primitives"]:
                prim["attributes"] = {n: a + acc0 for n, a in prim["attributes"].items()}
                if "indices" in prim:
                    prim["indices"] += acc0
                if "material" in prim:
                    prim["material"] = mat_map[prim["material"]]
                for target in prim.get("targets", []):
                    for n in target:
                        target[n] += acc0
            doc["meshes"].append(mesh)
        for node in c.get("nodes", []):
            node.pop("camera", None)
            if "mesh" in node:
                node["mesh"] += mesh0
            if "children" in node:
                node["children"] = [n + node0 for n in node["children"]]
            ext = node.get("extensions", {}).get("KHR_lights_punctual")
            if ext:
                ext["light"] = light_map[ext["light"]]
            doc["nodes"].append(node)
        scene = c["scenes"][c.get("scene", 0)] if c.get("scenes") else {"nodes": []}
        doc["scenes"][0]["nodes"] += [n + node0 for n in scene.get("nodes", [])]
    if lights:
        doc["extensions"] = {"KHR_lights_punctual": {"lights": lights}}
    for key in ("extensionsUsed", "extensionsRequired"):
        names = sorted(used if key == "extensionsUsed" else required)
        if names:
            doc[key] = names
    for key in [k for k, v in doc.items() if v == []]:
        del doc[key]
    tmp_out = os.path.join(out_dir, f".partial-{os.path.basename(out)}")
    with open(tmp_out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    shutil.rmtree(buf_dir, ignore_errors=True)
    os.replace(tmp_buf, buf_dir)
    os.replace(tmp_out, out)
    missing = [i["uri"] for i in doc.get("images", []) if "uri" in i
               and not os.path.exists(os.path.join(out_dir, urllib.parse.unquote(i["uri"])))]
    if missing:
        raise Failed(f"merged glTF points at missing images: {missing[:5]}")
    size = sum(os.path.getsize(os.path.join(buf_dir, n)) for n in os.listdir(buf_dir))
    tris = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            if "indices" in prim:
                tris += doc["accessors"][prim["indices"]]["count"] // 3
    print(f"WROTE {out}: {len(doc.get('nodes', []))} nodes, {len(doc.get('meshes', []))} meshes, ~{tris} unique "
          f"triangles, {len(doc.get('materials', []))} materials, {len(doc.get('images', []))} images, "
          f"{len(lights)} lights, buffers {size / 1e6:.0f} MB in {os.path.basename(buf_dir)}/", flush=True)
    if dropped:
        print(f"LOST textures on a UV map the mesh lacks (exporter wrote texCoord -1), removed [{len(dropped)}: "
              f"{', '.join(dropped[:6])}{' ...' if len(dropped) > 6 else ''}]", flush=True)
    if orm_occlusion:
        print("NOTE occlusion: each material's metallic-roughness texture is also its occlusionTexture (red)",
              flush=True)


# --- driver ------------------------------------------------------------------------------------------------------


def parse(argv):
    if "--" not in argv:
        raise SystemExit("usage: bounded.py --script X.py --output OUT.gltf [options] -- <script args> (see --help)")
    split = argv.index("--")
    own, script_args = argv[:split], argv[split + 1 :]
    opts = {"memory": DEFAULT_MEMORY, "chunk_tris": DEFAULT_CHUNK_TRIS, "keep_work": False, "orm_occlusion": False,
            "blender": os.environ.get("BLENDER") or shutil.which("blender") or os.path.expanduser("~/.local/bin/blender")}
    i = 0
    while i < len(own):
        name = own[i].lstrip("-").replace("-", "_")
        if name in ("keep_work", "orm_occlusion"):
            opts[name] = True
            i += 1
        else:
            opts[name] = own[i + 1]
            i += 2
    opts["chunk_tris"] = int(opts["chunk_tris"])
    if "script" not in opts or "output" not in opts or not opts["output"].endswith(".gltf"):
        raise SystemExit("bounded.py: --script and --output OUT.gltf are required")
    opts["output"] = os.path.abspath(opts["output"])
    return opts, script_args


def main(argv):
    if argv[:1] == ["_measure"]:
        return measure(argv[1], argv[3:])
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    opts, script_args = parse(argv)
    out = opts["output"]
    stem = os.path.splitext(os.path.basename(out))[0]
    work = os.path.join(os.path.dirname(out), f".work-{stem}")
    os.makedirs(work, exist_ok=True)
    opts.setdefault("log", os.path.join(os.path.dirname(out), f"convert-{stem}.log"))
    run = Runner(opts, work)
    stage = os.path.join(work, "stage")
    done = {"stage": False, "chunks": 0, "of": "?", "splittable": False}
    try:
        # 1. stage (redone when the converter's arguments change)
        key = json.dumps([opts["script"], script_args])
        stamp = os.path.join(stage, "args.json")
        staged = os.path.exists(os.path.join(stage, "scene.blend")) and os.path.exists(stamp) \
            and open(stamp).read() == key
        if staged:
            print(f"stage: done before ({stage})", flush=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
            open(opts["log"], "w").close()  # a fresh conversion starts a fresh log; a resumed one appends
            run.blender("stage", opts["script"], [*script_args, "--output", out, "--stage-dir", stage])
            with open(stamp, "w") as f:
                f.write(key)
        done["stage"] = True
        # 2. chunks (a plan is named by the staged objects and the chunk size; its finished chunks are reused)
        with open(os.path.join(stage, "objects.json")) as f:
            objects = json.load(f)
        chunks = plan_chunks(objects, opts["chunk_tris"])
        plan_id = hashlib.sha1(json.dumps([key, chunks]).encode()).hexdigest()[:8]
        done["of"] = len(chunks)
        chunk_dirs = []
        for n, names in enumerate(chunks):
            cdir = os.path.join(work, f"chunk-{plan_id}-{n:02}")
            chunk_dirs.append(cdir)
            if os.path.exists(os.path.join(cdir, "chunk.gltf")):
                done["chunks"] += 1
                continue
            tmp = cdir + ".partial"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            with open(os.path.join(tmp, "objects.json"), "w") as f:
                json.dump(names, f)
            meshes = {o["data"] for o in objects if o["name"] in set(names) and o["type"] == "MESH"}
            done["splittable"] = len(meshes) > 1
            run.blender(f"chunk {n + 1}/{len(chunks)}", "export_chunk.py",
                        ["--blend", os.path.join(stage, "scene.blend"), "--objects", os.path.join(tmp, "objects.json"),
                         "--output", os.path.join(tmp, "chunk.gltf")])
            os.replace(tmp, cdir)
            done["chunks"] += 1
        # 3. merge
        merge(chunk_dirs, out, opts["orm_occlusion"])
    except Killed as e:
        if done["splittable"]:
            fit = f"--chunk-tris {opts['chunk_tris'] // 2} (smaller chunks) or --memory larger than {opts['memory']}"
        else:
            what = "this chunk, a single mesh," if done["stage"] else "the stage step"
            fit = f"--memory larger than {opts['memory']} ({what} cannot be split)"
        print(f"KILLED: step {e} went over its memory cap ({opts['memory']}) and was stopped by it; nothing else "
              f"was touched. Kept in {work}: stage {'done' if done['stage'] else 'not done'}, chunks "
              f"{done['chunks']}/{done['of']}. Rerun the same command to resume. To fit: {fit}. Log: {opts['log']}",
              file=sys.stderr, flush=True)
        return 3
    except MemoryError as e:
        print(f"NOT STARTED: {e}", file=sys.stderr, flush=True)
        return 75
    except Failed as e:
        print(f"FAILED: {e}", file=sys.stderr, flush=True)
        return 1
    if run.peaks:
        print(f"PEAK max {max(run.peaks.values())} MB over {len(run.peaks)} Blender steps this run", flush=True)
    if not opts["keep_work"]:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
