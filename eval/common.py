"""Shared pieces of the smartbai evaluation (EVAL_PROTOCOL.md): case state, candidate actions, protocol evaluation."""
import json
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
from line_sim import LineSim, load_climate  # noqa: E402

HORIZON_H = 12.0
REPS = 8


@lru_cache(maxsize=1)
def load_inputs(config_path, climate_path):
    return json.load(open(config_path)), load_climate(climate_path)


def build_state(spec, cfg, climate):
    """twin state at the decision time of a case: warm-up, then the injected events (deterministic)."""
    sim = LineSim(cfg, climate, start=spec["start"], seed=int(spec["seed"]), params=spec.get("params"))
    sim.run(float(spec.get("warmup_h", 24)) * 60)
    for ev in spec.get("inject", []):
        sim.inject(ev)
    return sim


def evaluate(sim, actions, hours=HORIZON_H, reps=REPS):
    """net value (and KPIs) of an action list over the horizon; replications use common random numbers."""
    w = sim.whatif(actions, hours, reps=reps)
    return dict(net=[r["net_value"] for r in w["runs"]], mean=w["mean"])


def candidates(cfg):
    """the fixed candidate list of the protocol (same for every case)."""
    st = cfg["stations"]
    ids = [s["id"] for s in st]
    groups = {s["id"]: s["parallel_group"] for s in st}
    par = [s for s in ids if len(groups[s]) > 1]
    trunk = [x for g in cfg["lanes"]["L3"]["stages"] for x in g if x not in ("S31", "S32", "S38")]
    maint = lambda s: dict(action="schedule_maintenance", station_id=s, in_minutes=0.0)  # noqa: E731
    avoid = lambda s: dict(action="set_dispatch", weights={g: (0.0 if g == s else 1.0) for g in groups[s]})  # noqa: E731
    cool = dict(action="set_cooling", level="high")
    c = [("no_action", [])]
    c += [(f"maint_{s}", [maint(s)]) for s in ids]
    c += [(f"avoid_{s}", [avoid(s)]) for s in par]
    c += [(f"avoid4h_{s}", [dict(avoid(s), hours=4.0)]) for s in par]  # amendment 1
    c += [("cooling_high", [cool])]
    c += [("speed_trunk_115", [dict(action="set_speed", station_id=s, pct=115) for s in trunk]),
          ("speed_all_115", [dict(action="set_speed", station_id=s, pct=115) for s in ids]),
          ("speed_all_85", [dict(action="set_speed", station_id=s, pct=85) for s in ids])]
    c += [(f"maint_avoid_{s}", [maint(s), avoid(s)]) for s in par]
    c += [(f"maint_{s}_cooling", [maint(s), cool]) for s in ids]
    return c


def acceptable_set(table):
    """table: {name: [net per replication]} -> best name, and per name: paired shortfall, its SE, acceptable flag."""
    means = {k: float(np.mean(v)) for k, v in table.items()}
    best = max(means, key=means.get)
    out = {}
    for k, v in table.items():
        d = np.asarray(table[best]) - np.asarray(v)
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        tol = max(2 * se, 0.01 * abs(means[best]))
        out[k] = dict(mean=round(means[k], 1), shortfall=round(float(d.mean()), 1), se=round(se, 1), acceptable=bool(d.mean() <= tol))
    return best, out
