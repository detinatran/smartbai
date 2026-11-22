"""Build the dev / calibration / test cases of CERTIFY_PROTOCOL.md (deterministic: every draw of a case comes from its seed).
usage: scenarios_certify.py TWIN_V2_CONFIG CLIMATE_CSV OUT_JSON [--workers 8]"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GENERIC, HORIZON_H, P_SHORT, REPS, RUSH, build_state, evaluate, load_inputs  # noqa: E402
from line_sim import ASSUMED, LineSim  # noqa: E402

FAMILIES = ["degradation", "breakdown", "quality", "heat", "rush", "quiet"]
SPLITS = [("dev", 6, 20000), ("calibration", 35, 30000), ("test", 35, 50000)]
# amendment 3 (v2): fresh calibration and test seeds; the v1 splits were run with the single-stream simulator
SPLITS_V2 = [("dev", 6, 20000), ("calibration", 35, 60000), ("test", 35, 70000)]
# amendment 4 (v4): fresh calibration and test seeds for the new LTT test order; same simulator and generator as v2
SPLITS_V4 = [("dev", 6, 20000), ("calibration", 35, 80000), ("test", 35, 90000)]
PLANT_KEYS = ["mtbf_h", "mttr_min", "temp_k", "hall_coupling", "precursor_h", "degraded_defect_mult", "maint_min", "service_cv",
              "hvac_kw_per_deg", "cooling_high_coupling"]
FACTOR = 1.5
LEAD = dict(degradation=90.0, quality=180.0)  # amendment 1: minutes before the decision time
WARMUP_H = 24.0


def make_case(job):
    split, fam, k, seed, config, climate_csv = job
    cfg, climate = load_inputs(config, climate_csv)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-05-01") + pd.Timedelta(days=int(rng.integers(0, 153)), hours=int(rng.integers(6, 19)))
    params = {key: float(ASSUMED[key] * np.exp(rng.uniform(-np.log(FACTOR), np.log(FACTOR)))) for key in PLANT_KEYS}
    params["repair_est_sigma"] = 0.3
    spec = dict(case_id=f"{split}_{fam}_{k}", split=split, family=fam, start=str(start), seed=int(seed), warmup_h=WARMUP_H,
                params=params, twin=dict(forecast_bias=float(rng.normal(0.0, 1.5))), inject=[])
    st = cfg["stations"]
    ids = [s["id"] for s in st]
    share = np.array([s["share_parts"] for s in st])
    par = [s["id"] for s in st if len(s["parallel_group"]) > 1]
    end = WARMUP_H * 60.0
    if fam in LEAD:  # target drawn among stations running and not degraded when the event starts
        pre = LineSim(cfg, climate, start=spec["start"], seed=spec["seed"], params=params).run(end - LEAD[fam])
        w = share * np.array([pre.S[s]["state"] == "up" and not pre.S[s]["degraded"] for s in ids])
        sid = str(rng.choice(ids, p=w / w.sum()))
        if fam == "degradation":
            spec["inject"] = [dict(type="degrade", station=sid, lead_min=LEAD[fam])]
        else:
            spec["inject"] = [dict(type="defect_drift", station=sid, mult=float(rng.uniform(3, 8)), lead_min=LEAD[fam])]
    elif fam == "breakdown":
        pre = LineSim(cfg, climate, start=spec["start"], seed=spec["seed"], params=params).run(end)
        cand = [s for s in par if pre.S[s]["state"] == "up"]
        spec["inject"] = [dict(type="breakdown", station=str(rng.choice(cand)), repair_min=float(rng.uniform(120, 360)))]
    elif fam == "heat":
        spec["inject"] = [dict(type="heat", delta_c=float(rng.uniform(3, 8)), hours=HORIZON_H)]
    sim = build_state(spec, cfg, climate)
    spec["task"] = GENERIC.format(time=sim.clock())
    if fam == "rush":
        base = evaluate(sim, [], HORIZON_H, REPS)["mean"]["completed"]
        n = int(round(float(rng.uniform(1.02, 1.08)) * base))
        spec["inject"] = [dict(type="rush", target_parts=n, hours=HORIZON_H)]
        spec["task"] += RUSH.format(n=n, c=P_SHORT)
    spec["decision_time"] = str(sim.clock())
    return spec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("climate")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--version", choices=["v1", "v2", "v4"], default="v4")
    a = ap.parse_args()
    splits = dict(v1=SPLITS, v2=SPLITS_V2, v4=SPLITS_V4)[a.version]
    jobs = []
    for split, n, base in splits:
        for f_i, fam in enumerate(FAMILIES):
            for k in range(n):
                jobs.append((split, fam, k, base + f_i * 1000 + k, a.config, a.climate))
    with ProcessPoolExecutor(a.workers) as ex:
        cases = list(ex.map(make_case, jobs, chunksize=4))
    json.dump(dict(protocol="CERTIFY_PROTOCOL.md", version=a.version, horizon_h=HORIZON_H, plant_factor=FACTOR, cases=cases), open(a.out, "w"), indent=1)
    print({s: sum(c["split"] == s for c in cases) for s, _, _ in splits}, "cases")
