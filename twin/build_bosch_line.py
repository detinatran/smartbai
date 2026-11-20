"""smartbai twin v2: build a digital-twin scene whose topology and station parameters come from the Bosch Production
Line Performance data (config from data_tools/make_bosch_twin_config.py). Headless Isaac Sim; saves USD and renders.
Visual encodings: machine colour = failure rate relative to the line average (teal < 1.1x, yellow 1.1-1.5x, orange 1.5-3x,
red > 3x); sensor pucks on the head = number of numeric sensors of the station (capped at 12); parts on a belt are
proportional to the share of parts visiting that station; optional/side stations are lower.
usage: python build_bosch_line.py CONFIG_JSON OUTDIR"""
import json
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
cfg_path, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(cfg_path))
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
CONV_URL = f"{ROOT}/Isaac/Props/Conveyors/ConveyorBelt_A08.usd"
log = lambda *a: print("[v2]", *a, flush=True)  # noqa: E731

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
bbc = lambda: UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])  # noqa: E731


def xform(prim, pos=(0, 0, 0), rot_z=0.0, scale=1.0, rot_xyz=None):
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


def rng(prim):
    r = bbc().ComputeWorldBound(prim).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def box(path, center, size, color):
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


# ---------------------------------------------------------------- conveyor segment geometry (measured once)
probe = stage.DefinePrim("/World/_probe", "Xform")
probe.GetReferences().AddReference(CONV_URL)
for _ in range(10):
    app.update()
lo, hi = rng(probe)
SEG = hi[0] - lo[0]
surf = [rng(q)[1][2] - lo[2] for q in Usd.PrimRange(probe) if "roller" in q.GetName().lower() and q.IsA(UsdGeom.Gprim)]
ZB = max(surf) if surf else 0.77
OFF = ((lo + hi) / 2)
stage.RemovePrim("/World/_probe")
log(f"segment {SEG:.3f} m, belt surface {ZB:.3f} m")
n_conv = 0


def conveyor(x, y, rot=0):
    global n_conv
    p = stage.DefinePrim(f"/World/Conveyors/c_{n_conv:03d}", "Xform")
    p.GetReferences().AddReference(CONV_URL)
    n_conv += 1
    if rot == 0:
        xform(p, (x - OFF[0], y - OFF[1], -lo[2]))
    else:  # 90 deg about z: local (u, v) -> (-v, u)
        xform(p, (x + OFF[1], y - OFF[0], -lo[2]), rot_z=90)
    return p


# ---------------------------------------------------------------- layout
TRACK = 1.55
LANE_Y = {"L0A": 7.0, "L0B": 0.0, "L1-L2": -7.0}
N_FEED = max(len(v["stages"]) for k, v in CFG["lanes"].items() if k != "L3")
MERGE_X = N_FEED * SEG
st_by_id = {s["id"]: s for s in CFG["stations"]}
pos = {}
for lane, d in CFG["lanes"].items():
    for k, grp in enumerate(d["stages"]):
        if lane == "L3":
            x, yc = MERGE_X + 1.95 + k * SEG, 0.0   # trunk starts right after the collector
        else:
            x, yc = (N_FEED - len(d["stages"]) + k) * SEG, LANE_Y[lane]
        for j, sid in enumerate(grp):
            pos[sid] = (x + SEG / 2, yc + (j - (len(grp) - 1) / 2) * TRACK)
# conveyors under every station, transfer plates between stages, a collector along y at the merge
for lane, d in CFG["lanes"].items():
    prev = None
    for grp in d["stages"]:
        ys = [pos[s][1] for s in grp]
        for s in grp:
            conveyor(pos[s][0], pos[s][1])
        if prev is not None:
            ally = prev + ys
            xb = pos[grp[0]][0] - SEG / 2
            box(f"/World/Transfers/t_{lane}_{xb:.1f}".replace(".", "_").replace("-", "m"),
                (xb, (min(ally) + max(ally)) / 2, ZB - 0.03), (0.35, max(ally) - min(ally) + 1.0, 0.06), (0.55, 0.57, 0.6))
        prev = ys
ys_feed = sorted(LANE_Y.values())
for k in range(int((ys_feed[-1] - ys_feed[0]) / SEG) + 1):
    conveyor(MERGE_X + SEG / 2, ys_feed[0] + k * SEG + SEG / 2 - 0.4, rot=90)
box("/World/Transfers/merge_in", (MERGE_X + 1.95, 0, ZB - 0.03), (0.35, 2.5, 0.06), (0.55, 0.57, 0.6))
for lane, yc in LANE_Y.items():  # each feeder lane hands over to the collector
    last = [pos[s][1] for s in CFG["lanes"][lane]["stages"][-1]]
    box(f"/World/Transfers/to_collector_{lane.replace('-', '_')}", (MERGE_X + 0.4, (min(last) + max(last)) / 2, ZB - 0.03), (0.8, max(last) - min(last) + 1.0, 0.06), (0.55, 0.57, 0.6))

# ---------------------------------------------------------------- machines
avg = CFG["fail_rate"]


def colour(f):
    r = f / avg
    return (0.15, 0.62, 0.6) if r < 1.1 else (0.95, 0.8, 0.2) if r < 1.5 else (0.95, 0.5, 0.12) if r < 3 else (0.85, 0.12, 0.1)


for s in CFG["stations"]:
    x, y = pos[s["id"]]
    root = f"/World/Stations/{s['id']}"
    p = UsdGeom.Xform.Define(stage, root).GetPrim()
    for k in ("id", "lane", "parallel_group"):
        p.CreateAttribute(f"smartbai:{k}", Sdf.ValueTypeNames.String).Set(str(s[k]) if k != "parallel_group" else ",".join(s[k]))
    for k in ("share_parts", "fail_rate", "median_offset"):
        p.CreateAttribute(f"smartbai:{k}", Sdf.ValueTypeNames.Float).Set(float(s[k]))
    for k in ("line", "stage", "n_sensors", "parts"):
        p.CreateAttribute(f"smartbai:{k}", Sdf.ValueTypeNames.Int).Set(int(s[k]))
    p.CreateAttribute("smartbai:optional", Sdf.ValueTypeNames.Bool).Set(bool(s["optional"]))
    h = 1.35 if s["optional"] else 1.9
    c = colour(s["fail_rate"])
    for side in (-1, 1):
        box(f"{root}/pillar_{'L' if side < 0 else 'R'}", (x, y + side * 0.62, h / 2), (0.8, 0.18, h), c)
    box(f"{root}/head", (x, y, h - 0.22), (0.8, 1.42, 0.44), c)
    n = min(s["n_sensors"], 12)
    for i in range(n):  # sensor pucks in two rows on the head
        cyl(f"{root}/sensor_{i:02d}", (x - 0.3 + (i % 6) * 0.12, y + (-0.18 if i < 6 else 0.18), h + 0.02), 0.04, 0.04, (0.12, 0.12, 0.14))
    sphere(f"{root}/status_lamp", (x + 0.32, y - 0.62, h + 0.12), 0.07, (0.1, 0.9, 0.2)).GetPrim().CreateAttribute(
        "smartbai:status", Sdf.ValueTypeNames.String).Set("running")
    for i in range(max(1, round(s["share_parts"] * 6))):  # parts on the belt ~ flow share
        box(f"{root}/part_{i}", (x - SEG / 2 + 0.35 + i * 0.32, y, ZB + 0.04), (0.16, 0.11, 0.08), (0.62, 0.64, 0.68))

# ---------------------------------------------------------------- floor, lights
x_max = max(v[0] for v in pos.values()) + SEG
box("/World/Floor", (x_max / 2, 0, -0.01), (x_max + 12, 30, 0.02), (0.2, 0.22, 0.25))
for lane, yc in LANE_Y.items():  # floor stripe per feeder lane
    box(f"/World/LaneMark_{lane.replace('-', '_')}", (MERGE_X / 2, yc, 0.002), (MERGE_X + 1, 2.2 + 1.55 * 2, 0.004), (0.27, 0.29, 0.33))
UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(600)
key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
key.CreateIntensityAttr(2200)
xform(key.GetPrim(), (0, 0, 20), rot_xyz=(-45, 15, 30))
for _ in range(60):
    app.update()
usd_path = os.path.join(OUT, "smartbai_twin_v2.usd")
ctx.save_as_stage(usd_path)
stage = ctx.get_stage()
log("saved", usd_path, "conveyor segments:", n_conv)

# ---------------------------------------------------------------- renders
import omni.replicator.core as rep  # noqa: E402

xs32, ys32 = pos["S32"]
x35, _ = pos["S35"]
views = dict(
    overview=dict(position=(x_max / 2 - 2, -26, 20), look_at=(x_max / 2, 0, 0), focal=16),
    feeders=dict(position=(-6, -14, 9), look_at=(10, 0, 0.5), focal=16),
    trunk_hotspot=dict(position=(xs32 - 4, -6.5, 4.0), look_at=(xs32 + 1.5, 0, 0.8), focal=16),
    parallel_s35_s36=dict(position=(x35 + 4.5, -5.0, 3.6), look_at=(x35, 0, 0.8), focal=18),
    top=dict(position=(x_max / 2, 0, 48), look_at=(x_max / 2, 0.01, 0), focal=14),
)
rps = {}
for name, v in views.items():
    cam = rep.create.camera(position=v["position"], look_at=v["look_at"], focal_length=v["focal"])
    rps[name] = rep.create.render_product(cam, (1600, 900))
for name, rp in rps.items():
    w = rep.WriterRegistry.get("BasicWriter")
    w.initialize(output_dir=os.path.join(OUT, name), rgb=True)
    w.attach([rp])
rep.orchestrator.step(rt_subframes=32)
rep.orchestrator.wait_until_complete()
json.dump(dict(positions=pos, segment_m=SEG, belt_surface_m=ZB, merge_x=MERGE_X, seconds=round(time.time() - t0)),
          open(os.path.join(OUT, "scene_meta.json"), "w"), indent=1)
log("DONE", round(time.time() - t0), "s")
app.close()
