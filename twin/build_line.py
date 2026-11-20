"""smartbai: build the bottling-line digital twin scene in Isaac Sim (headless), save it as USD and render views.
usage: python build_line.py OUTDIR [--no-env]"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from line_config import CONVEYOR_LENGTH_M, LINE, STATIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--no-env", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1600, "height": 900})
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

try:
    from isaacsim.storage.native import get_assets_root_path
except ImportError:
    from isaacsim.core.utils.nucleus import get_assets_root_path
ROOT = get_assets_root_path()
A = lambda p: f"{ROOT}/Isaac/{p}"  # noqa: E731
log = lambda *a: print("[line]", *a, flush=True)  # noqa: E731

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
bbc = lambda: UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])  # noqa: E731


def xform(prim, pos=(0, 0, 0), rot_z=0.0, scale=1.0, rot_xyz=None):
    """own translate/rotate/scale ops (suffix 'sb'); replaces any op order authored inside a referenced asset."""
    xf = UsdGeom.Xformable(prim)
    if not prim.HasAttribute("xformOp:translate:sb"):
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "sb")
        xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat, "sb")
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "sb")
    prim.GetAttribute("xformOp:translate:sb").Set(Gf.Vec3d(*pos))
    prim.GetAttribute("xformOp:rotateXYZ:sb").Set(Gf.Vec3f(*(rot_xyz or (0, 0, rot_z))))
    s = scale if isinstance(scale, tuple) else (scale, scale, scale)
    prim.GetAttribute("xformOp:scale:sb").Set(Gf.Vec3f(*s))


def ref(path, url, pos=(0, 0, 0), rot_z=0.0, scale=1.0):
    p = stage.DefinePrim(path, "Xform")
    p.GetReferences().AddReference(url)
    xform(p, pos, rot_z, scale)
    return p


def rng(prim):
    r = bbc().ComputeWorldBound(prim).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def box(path, center, size, color, rough=0.6):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xform(c.GetPrim(), center, 0, tuple(size))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return c


def cyl(path, center, radius, height, color):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr("Z")
    xform(c.GetPrim(), center)
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return c


def sphere(path, center, radius, color):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(radius)
    xform(s.GetPrim(), center)
    s.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return s


# ---------------------------------------------------------------- environment and light
if not args.no_env:
    # interior of Simple_Warehouse: x in [-9.9, 8.9], y in [-9.6, 20.8] (probe_walls.py); rotate so its 30 m axis runs along the line
    env = ref("/World/Env", A("Environments/Simple_Warehouse/warehouse.usd"), pos=(16.5, 0, 0), rot_z=90)
    for _ in range(20):
        app.update()
    lo, hi = rng(env)
    log("warehouse bbox", tuple(round(v, 2) for v in lo), tuple(round(v, 2) for v in hi))
    if (hi - lo).GetLength() < 1e-3:
        log("warehouse failed to load; continuing without it")
        stage.RemovePrim("/World/Env")
UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(600)
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
sun.CreateIntensityAttr(2500)
xform(sun.GetPrim(), (0, 0, 10), rot_xyz=(-40, 20, 30))

# ---------------------------------------------------------------- conveyor (tiled NVIDIA conveyor segment)
CONV_URL = A("Props/Conveyors/ConveyorBelt_A08.usd")
probe = ref("/World/_probe", CONV_URL)
for _ in range(10):
    app.update()
lo, hi = rng(probe)
size = hi - lo
log("conveyor segment bbox size", tuple(round(v, 3) for v in size))
# belt surface = top of the rollers/belt meshes, not the bbox top (which includes sensor posts and guards)
surf = []
for key in ("roller", "belt"):
    for q in Usd.PrimRange(probe):
        if key in q.GetName().lower() and q.IsA(UsdGeom.Gprim):
            surf.append(rng(q)[1][2] - lo[2])
    if surf:
        break
log("belt-surface candidates:", len(surf), "max", round(max(surf), 3) if surf else None)
stage.RemovePrim("/World/_probe")
unit = 0.01 if max(size) > 50 else 1.0          # some assets are authored in cm
size = size * unit
along_x = size[0] >= size[1]
seg_len = size[0] if along_x else size[1]
belt_top = (max(surf) * unit) if surf else 0.72 * size[2]
n_seg = max(1, math.ceil(CONVEYOR_LENGTH_M / seg_len))
for i in range(n_seg):
    x = i * seg_len + seg_len / 2 - 0.0
    p = ref(f"/World/Conveyor/seg_{i:02d}", CONV_URL, pos=(0, 0, 0), rot_z=0 if along_x else 90, scale=unit)
    l2, h2 = rng(p)                               # recentre each segment on (x, 0) with its base on the floor
    c = (l2 + h2) / 2
    xform(p, (x - c[0], -c[1], -l2[2]), 0 if along_x else 90, unit)
log(f"conveyor: {n_seg} segments x {seg_len:.2f} m, belt top ~{belt_top:.2f} m")
z_belt = belt_top

# ---------------------------------------------------------------- stations
COL = dict(filler=(0.75, 0.78, 0.82), capper=(0.2, 0.45, 0.75), labeler=(0.9, 0.62, 0.1), vision_qc=(0.25, 0.25, 0.28))
for s in STATIONS:
    root = f"/World/Stations/{s['id']}_{s['name'].replace(' ', '_')}"
    st = UsdGeom.Xform.Define(stage, root).GetPrim()
    for k, v in (("station_id", s["id"]), ("name", s["name"]), ("type", s["type"]), ("sensors", ",".join(s["sensors"]))):
        st.CreateAttribute(f"smartbai:{k}", Sdf.ValueTypeNames.String).Set(v)
    st.CreateAttribute("smartbai:rate_bpm", Sdf.ValueTypeNames.Float).Set(float(s["rate_bpm"]))
    x = s["x"]
    if s["type"].startswith("robot"):
        robot = "ur10" if s["type"] == "robot_ur10" else "ur5e"
        side = -1.0
        box(f"{root}/pedestal", (x, side * 0.9, 0.4), (0.5, 0.5, 0.8), (0.3, 0.3, 0.32))
        ref(f"{root}/robot", A(f"Robots/UniversalRobots/{robot}/{robot}.usd"), pos=(x, side * 0.9, 0.8), rot_z=90)
        if s["id"] in ("S1", "S7"):              # pallets with cartons on the far side
            ref(f"{root}/pallet", A("Props/Pallet/pallet.usd"), pos=(x, side * 2.2, 0))
            for i in range(3):
                for j in range(2):
                    for k in range(2 if s["id"] == "S1" else 1):
                        box(f"{root}/carton_{i}{j}{k}", (x - 0.4 + i * 0.4, side * 2.2 - 0.25 + j * 0.5, 0.35 + k * 0.32),
                            (0.38, 0.48, 0.3), (0.72, 0.56, 0.36))
        else:                                     # case packer: open cartons on a table
            box(f"{root}/table", (x, 1.0, 0.45), (1.2, 0.7, 0.9), (0.55, 0.55, 0.58))
            for i in range(2):
                box(f"{root}/case_{i}", (x - 0.3 + i * 0.6, 1.0, 1.05), (0.45, 0.35, 0.3), (0.72, 0.56, 0.36))
        lamp_pos = (x, side * 0.9 + 0.3, 0.95)
    else:                                         # gantry-style machine housing spanning the belt
        c = COL[s["type"]]
        h = 2.0
        for side in (-1, 1):
            box(f"{root}/pillar_{'L' if side < 0 else 'R'}", (x, side * 0.65, h / 2), (0.9, 0.25, h), c)
        box(f"{root}/head", (x, 0, h - 0.25), (0.9, 1.55, 0.5), c)
        box(f"{root}/hmi", (x + 0.5, -0.8, 1.3), (0.05, 0.35, 0.28), (0.05, 0.05, 0.08))
        if s["type"] == "filler":
            for i in range(4):
                cyl(f"{root}/nozzle_{i}", (x - 0.3 + i * 0.2, 0, h - 0.6), 0.03, 0.3, (0.85, 0.85, 0.9))
        if s["type"] == "vision_qc":
            ref(f"{root}/camera_prop", A("Props/Camera/camera.usd"), pos=(x, 0, h - 0.55))
            qc = UsdGeom.Camera.Define(stage, f"{root}/qc_camera")
            xform(qc.GetPrim(), (x, 0, h - 0.52))       # just under the machine head, looking straight down (-Z)
            qc.CreateFocalLengthAttr(10.0)
            qc.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
            box(f"{root}/reject_bin", (x + 0.6, 0.75, 0.3), (0.5, 0.4, 0.6), (0.7, 0.15, 0.12))
        lamp_pos = (x, -0.65, h + 0.15)
    lamp = sphere(f"{root}/status_lamp", lamp_pos, 0.08, (0.1, 0.9, 0.2))
    lamp.GetPrim().CreateAttribute("smartbai:status", Sdf.ValueTypeNames.String).Set("running")

# ---------------------------------------------------------------- bottles: state follows the stations they passed
xs = {s["type"]: s["x"] for s in STATIONS}
n = int((STATIONS[-2]["x"] - STATIONS[0]["x"]) / LINE["bottle_pitch_m"])
for i in range(n):
    x = STATIONS[0]["x"] + 0.6 + i * LINE["bottle_pitch_m"]
    if x > STATIONS[-2]["x"] - 0.5:
        break
    b = f"/World/Bottles/b_{i:03d}"
    filled, capped, labelled = x > xs["filler"], x > xs["capper"], x > xs["labeler"]
    body = (0.95, 0.55, 0.12) if filled else (0.78, 0.88, 0.95)
    cyl(f"{b}/body", (x, 0, z_belt + 0.11), 0.033, 0.22, body)
    cyl(f"{b}/neck", (x, 0, z_belt + 0.24), 0.014, 0.04, (0.78, 0.88, 0.95))
    if capped:
        cyl(f"{b}/cap", (x, 0, z_belt + 0.27), 0.016, 0.02, (0.85, 0.1, 0.1))
    if labelled:
        cyl(f"{b}/label", (x, 0, z_belt + 0.10), 0.0345, 0.08, (0.1, 0.35, 0.75))

for _ in range(60):  # let streamed assets and textures arrive
    app.update()

usd_path = os.path.join(args.out, "smartbai_line.usd")
ctx.save_as_stage(usd_path)
log("saved", usd_path)

# ---------------------------------------------------------------- render views
import omni.replicator.core as rep  # noqa: E402

views = dict(
    overview=dict(position=(1.0, -5.5, 6.5), look_at=(12.0, 0, 0.6)),
    side=dict(position=(10.5, -5.8, 2.6), look_at=(10.5, 0, 0.9)),
    filler_close=dict(position=(3.2, -2.6, 2.0), look_at=(6.5, 0, 1.0)),
    end_of_line=dict(position=(24.5, -4.0, 3.0), look_at=(16.0, 0, 0.8)),
)
rps = {}
for name, v in views.items():
    cam = rep.create.camera(position=v["position"], look_at=v["look_at"], focal_length=v.get("focal", 14))
    rps[name] = rep.create.render_product(cam, (1600, 900))
stage = ctx.get_stage()  # save_as_stage reopens the stage; the old handle is stale
qc = next(s for s in STATIONS if s["type"] == "vision_qc")
rps["qc_camera"] = rep.create.render_product(f"/World/Stations/{qc['id']}_{qc['name'].replace(' ', '_')}/qc_camera", (800, 600))
for name, rp in rps.items():
    w = rep.WriterRegistry.get("BasicWriter")
    w.initialize(output_dir=os.path.join(args.out, name), rgb=True)
    w.attach([rp])
rep.orchestrator.step(rt_subframes=32)
rep.orchestrator.wait_until_complete()
meta = dict(line=LINE, stations=STATIONS, conveyor=dict(segments=n_seg, segment_len_m=seg_len, belt_top_m=z_belt),
            assets_root=ROOT, seconds=round(time.time() - t0))
json.dump(meta, open(os.path.join(args.out, "scene_meta.json"), "w"), indent=1)
log("DONE", meta["seconds"], "s")
app.close()
