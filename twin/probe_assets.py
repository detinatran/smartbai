"""smartbai: check which NVIDIA Isaac assets are reachable (conveyors, props, environments)."""
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import omni.client  # noqa: E402

try:
    from isaacsim.storage.native import get_assets_root_path
except ImportError:
    from isaacsim.core.utils.nucleus import get_assets_root_path
root = get_assets_root_path()
print("[probe] assets root:", root, flush=True)


def ls(path, depth=0, maxd=1):
    res, entries = omni.client.list(path)
    if res != omni.client.Result.OK:
        print("[probe]", "  " * depth, "FAIL", path, res, flush=True)
        return
    for e in entries:
        print("[probe]", "  " * depth, e.relative_path, flush=True)
        if depth < maxd and not e.relative_path.endswith((".usd", ".usda", ".usdz", ".png", ".jpg", ".mdl")):
            ls(path.rstrip("/") + "/" + e.relative_path, depth + 1, maxd)


if root:
    for sub in ["Isaac/Props/Conveyors", "Isaac/Environments", "Isaac/Props", "Isaac/Robots/UniversalRobots"]:
        print("[probe] ==", sub, flush=True)
        ls(f"{root}/{sub}", 0, 0 if sub in ("Isaac/Props", "Isaac/Environments") else 1)
print("[probe] DONE", flush=True)
app.close()
