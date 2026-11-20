"""smartbai: print wall/floor prims of the Simple_Warehouse asset with world bounding boxes (to place the line inside)."""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
try:
    from isaacsim.storage.native import get_assets_root_path
except ImportError:
    from isaacsim.core.utils.nucleus import get_assets_root_path
ctx = omni.usd.get_context(); ctx.new_stage(); st = ctx.get_stage()
p = st.DefinePrim("/W", "Xform"); p.GetReferences().AddReference(f"{get_assets_root_path()}/Isaac/Environments/Simple_Warehouse/warehouse.usd")
for _ in range(20): app.update()
bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
rows = []
for q in Usd.PrimRange(p):
    n = q.GetName().lower()
    if q.IsA(UsdGeom.Gprim) or q.IsA(UsdGeom.Xform):
        if any(k in n for k in ("wall", "floor", "rack", "shelf", "pillar", "column")) and q.GetParent().GetName().lower() not in n:
            r = bc.ComputeWorldBound(q).ComputeAlignedRange()
            lo, hi = r.GetMin(), r.GetMax()
            rows.append((q.GetPath().pathString[-60:], [round(v, 1) for v in lo], [round(v, 1) for v in hi]))
for r in rows[:80]:
    print("[walls]", *r, flush=True)
print("[walls] total", len(rows), flush=True)
app.close()
