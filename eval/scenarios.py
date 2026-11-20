"""Build the 30 evaluation cases of EVAL_PROTOCOL.md (deterministic, seed 2026).
usage: scenarios.py TWIN_V2_CONFIG CLIMATE_CSV OUT_JSON"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HORIZON_H, REPS, build_state, evaluate, load_inputs  # noqa: E402

cfg, climate = load_inputs(sys.argv[1], sys.argv[2])
rng = np.random.default_rng(2026)
st = cfg["stations"]
ids = [s["id"] for s in st]
share = np.array([s["share_parts"] for s in st])
par = [s["id"] for s in st if len(s["parallel_group"]) > 1]
P_VALUE, P_SHORT = 10.0, 40.0

GENERIC = ("You are the supervisor of this production line. The current twin time is {time}. Use the tools to check the line, "
           "then take the actions you think are best for the next 12 hours (taking no action is allowed). Finally call "
           "submit_decision with your actions, your rationale and claims that cite observation_ids. Objective: maximise net "
           "value over the next 12 hours = value of completed parts minus energy, defect and unplanned-downtime costs.")
RUSH = " A customer needs at least {n} completed parts in the next 12 hours; every part short costs {c:.0f} units."

cases = []
families = ["degradation", "breakdown", "quality", "heat", "rush"]
for f_i, fam in enumerate(families):
    for k in range(6):
        i = f_i * 6 + k
        day = 1 + i                       # one July day per case
        spec = dict(case_id=f"{fam}_{k}", family=fam, start=f"2025-07-{day:02d} 10:00", seed=1000 + i, warmup_h=24, inject=[])
        # amendment 1: target stations are drawn among stations that are running (and not degraded) at the decision time
        pre = build_state(spec, cfg, climate)
        ok = np.array([pre.S[s]["state"] == "up" and not pre.S[s]["degraded"] for s in ids])
        w = share * ok
        if fam == "degradation":
            spec["inject"] = [dict(type="degrade", station=str(rng.choice(ids, p=w / w.sum())))]
        elif fam == "breakdown":
            cand = [s for s in par if pre.S[s]["state"] == "up"]
            spec["inject"] = [dict(type="breakdown", station=str(rng.choice(cand)), repair_min=float(rng.integers(180, 301)))]
        elif fam == "quality":
            spec["inject"] = [dict(type="defect_drift", station=str(rng.choice(ids, p=w / w.sum())), mult=6.0)]
        elif fam == "heat":
            spec["inject"] = [dict(type="heat", delta_c=6.0, hours=HORIZON_H)]
        sim = build_state(spec, cfg, climate)
        task = GENERIC.format(time=sim.clock())
        if fam == "rush":
            base = evaluate(sim, [], HORIZON_H, REPS)["mean"]["completed"]
            n = int(round(1.04 * base))
            spec["inject"] = [dict(type="rush", target_parts=n, hours=HORIZON_H)]
            task += RUSH.format(n=n, c=P_SHORT)
        spec["task"] = task
        spec["decision_time"] = str(sim.clock())
        cases.append(spec)
        print(spec["case_id"], spec["inject"], flush=True)
json.dump(dict(protocol="EVAL_PROTOCOL.md", horizon_h=HORIZON_H, reps=REPS, cases=cases), open(sys.argv[3], "w"), indent=1)
print(len(cases), "cases")
