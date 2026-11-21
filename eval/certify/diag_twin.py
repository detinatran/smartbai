"""Dev-only diagnostic of the twin-plant gap (not part of the confirmatory protocol).
For dev-seed cases of the given families, compares the twin and the plant under no action and under fixed probe
actions: completed parts, rush shortfall, cost and net value, and the paired gains twin vs plant.
Uses only dev seeds (20000+): the 36 dev cases of cases.json plus extra cases drawn with the same generator.
usage: diag_twin.py CONFIG CLIMATE CASES_JSON FAILURE_MODEL_JSON --families rush quiet --extra 24 --out OUT_JSONL"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim"))
from common import HORIZON_H, build_state, candidates, load_inputs  # noqa: E402
from scenarios_certify import FAMILIES, make_case  # noqa: E402
from twin_sync import TwinModel  # noqa: E402

REPS = 32
KEYS = ("completed", "shortfall_parts", "cost", "net_value")
PROBES = ("speed_trunk_115", "speed_all_115", "cooling_high", "maint_S30", "avoid_S16")


def summarize(runs):
    return {k: float(np.mean([r[k] for r in runs])) for k in KEYS}


def diag_case(job):
    spec, config, climate_csv, fm = job
    cfg, climate = load_inputs(config, climate_csv)
    plant = build_state(spec, cfg, climate)
    twin = TwinModel(plant, seed=[int(spec["seed"]), 7], failure_model=fm,
                     forecast_bias=float(spec.get("twin", {}).get("forecast_bias", 0.0)), horizon_h=HORIZON_H)
    cand = dict(candidates(cfg))
    out = dict(case_id=spec["case_id"], family=spec["family"], params={k: spec["params"][k] for k in ("mtbf_h", "mttr_min", "service_cv", "maint_min")},
               target=next((e["target_parts"] for e in spec["inject"] if e["type"] == "rush"), None))
    p0 = plant.whatif([], HORIZON_H, reps=REPS, seed_offset=1_000_000)["runs"]
    t0 = [twin.run([], r, HORIZON_H) for r in range(REPS)]
    out["plant_none"], out["twin_none"] = summarize(p0), summarize(t0)
    for name in PROBES:
        p1 = plant.whatif(cand[name], HORIZON_H, reps=REPS, seed_offset=1_000_000)["runs"]
        t1 = [twin.run(cand[name], r, HORIZON_H) for r in range(REPS)]
        dp = np.array([a["net_value"] - b["net_value"] for a, b in zip(p1, p0)])
        dt = np.array([a["net_value"] - b["net_value"] for a, b in zip(t1, t0)])
        out[name] = dict(plant_se=float(dp.std(ddof=1) / np.sqrt(REPS)), twin_se=float(dt.std(ddof=1) / np.sqrt(REPS)),
                         plant_gain={k: float(np.mean([a[k] - b[k] for a, b in zip(p1, p0)])) for k in KEYS},
                         twin_gain={k: float(np.mean([a[k] - b[k] for a, b in zip(t1, t0)])) for k in KEYS})
    print(spec["case_id"], round(out["twin_none"]["completed"] - out["plant_none"]["completed"]),
          {n: (round(out[n]["twin_gain"]["net_value"]), round(out[n]["plant_gain"]["net_value"])) for n in PROBES}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("climate")
    ap.add_argument("cases")
    ap.add_argument("failure_model")
    ap.add_argument("--families", nargs="+", default=["rush", "quiet"])
    ap.add_argument("--extra", type=int, default=0, help="extra dev-seed cases per family (k = 6 .. 6+extra-1)")
    ap.add_argument("--first", type=int, default=6, help="first k of the extra cases")
    ap.add_argument("--skip-existing", action="store_true", help="only the freshly generated cases")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fm = json.load(open(a.failure_model))
    specs = [] if a.skip_existing else [c for c in json.load(open(a.cases))["cases"] if c["split"] == "dev" and c["family"] in a.families]
    jobs = [("dev", f, k, 20000 + FAMILIES.index(f) * 1000 + k, a.config, a.climate) for f in a.families for k in range(a.first, a.first + a.extra)]
    with ProcessPoolExecutor(a.workers) as ex:
        specs += list(ex.map(make_case, jobs))
        assert all(s["split"] == "dev" and 20000 <= s["seed"] < 30000 for s in specs)
        res = list(ex.map(diag_case, [(s, a.config, a.climate, fm) for s in specs]))
    with open(a.out, "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
