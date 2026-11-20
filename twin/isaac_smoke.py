"""smartbai: Isaac Sim headless smoke test (physics + RTX render). EULA accepted by the user on 2026-11-20."""
import os
import sys
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/isaac_smoke"
t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 640, "height": 480})
print(f"[smoke] app started in {time.time() - t0:.0f}s", flush=True)

import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ctx = omni.usd.get_context()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdPhysics.Scene.Define(stage, "/World/physics").CreateGravityMagnitudeAttr(9.81)
ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.AddScaleOp().Set(Gf.Vec3f(5, 5, 0.05))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
cube = UsdGeom.Cube.Define(stage, "/World/box")
cube.AddTranslateOp().Set(Gf.Vec3f(0, 0, 1.5))
cube.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 0.2))
cube.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.3, 0.1)])
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(3000)

import omni.timeline  # noqa: E402

tl = omni.timeline.get_timeline_interface()
tl.play()
for _ in range(120):
    app.update()
z = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(0).ExtractTranslation()[2]
print(f"[smoke] box z after 120 steps: {z:.3f} (started 1.5; resting ~0.25 if physics works)", flush=True)

try:
    import omni.replicator.core as rep

    cam = rep.create.camera(position=(3, 3, 2), look_at=(0, 0, 0.3))
    rp = rep.create.render_product(cam, (640, 480))
    w = rep.WriterRegistry.get("BasicWriter")
    w.initialize(output_dir=OUT, rgb=True)
    w.attach([rp])
    for _ in range(3):
        rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()
    print(f"[smoke] replicator wrote: {sorted(os.listdir(OUT))[:5]}", flush=True)
except Exception as e:  # report, do not hide
    print(f"[smoke] replicator FAILED: {type(e).__name__}: {e}", flush=True)

tl.stop()
print(f"[smoke] DONE total {time.time() - t0:.0f}s", flush=True)
app.close()
