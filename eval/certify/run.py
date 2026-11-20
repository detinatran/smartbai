"""Run Certify over one split (CERTIFY_PROTOCOL.md): for every case and proposer, sequential paired testing in the twin,
then the plant harm label of the chosen candidate; plus the rule agent executed directly (condition "rules").
One JSON line per (case, proposer). Resumes: cases already in the output file are skipped.
usage: run.py CONFIG CLIMATE CASES_JSON FAILURE_MODEL_JSON --split dev --out OUT_JSONL [--llm-proposals P.jsonl] [--workers 8]"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim"))
from common import HORIZON_H, build_state, load_inputs  # noqa: E402
from proposers import random_proposals, rule_proposals  # noqa: E402
from rs import certify_rs  # noqa: E402
from twin_sync import TwinModel  # noqa: E402

PLANT_REPS, PLANT_OFFSET = 96, 1_000_000  # amendment 2
MARGIN = 0.01  # amendment 2: harm = plant gain below -1% of the plant's no-action net value


def plant_runs(plant, actions):
    w = plant.whatif(actions, HORIZON_H, reps=PLANT_REPS, seed_offset=PLANT_OFFSET)
    return np.array([r["net_value"] for r in w["runs"]])


def run_case(job):
    spec, config, climate_csv, fm, llm = job
    t0 = time.time()
    cfg, climate = load_inputs(config, climate_csv)
    plant = build_state(spec, cfg, climate)
    base = plant_runs(plant, [])
    margin = MARGIN * abs(float(base.mean()))
    cache = {}

    def delta(actions):
        key = json.dumps(actions, sort_keys=True)
        if key not in cache:
            d = plant_runs(plant, actions) - base
            cache[key] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))
        return cache[key]

    rows = []
    rule_c, rule_acts = rule_proposals(spec, cfg, climate)
    props = dict(rule=rule_c, random5=random_proposals(spec, cfg))
    if llm is not None:
        props["llm"] = llm.get("candidates", [])
    common = dict(case_id=spec["case_id"], split=spec["split"], family=spec["family"], plant_no_action=float(base.mean()), margin=margin)
    for name, cands in props.items():
        twin = TwinModel(plant, seed=[int(spec["seed"]), 7], failure_model=fm,
                         forecast_bias=float(spec.get("twin", {}).get("forecast_bias", 0.0)), horizon_h=HORIZON_H)
        r = certify_rs(twin, cands, hours=HORIZON_H)
        row = dict(common, proposer=name, candidates=cands, s=r["s"], j_star=r["j_star"], actions=r.get("actions"),
                   twin_gain=r.get("mean_gain"), twin_se=r.get("se"), twin_reps=r["used"], n=r["n"])
        if r["j_star"] is not None:
            row["plant_delta"], row["plant_se"] = delta(r["actions"])
            row["harm"] = int(row["plant_delta"] < -margin)
        rows.append(row)
    rd = dict(zip(("plant_delta", "plant_se"), delta(rule_acts))) if rule_acts else dict(plant_delta=0.0, plant_se=0.0)
    rows.append(dict(common, proposer="rules_direct", actions=rule_acts, s=None, harm=int(rd["plant_delta"] < -margin), **rd))
    for row in rows:
        row["sec"] = round(time.time() - t0, 1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("climate")
    ap.add_argument("cases")
    ap.add_argument("failure_model")
    ap.add_argument("--split", required=True, choices=["dev", "calibration", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-proposals")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    fm = json.load(open(a.failure_model))
    llm = {json.loads(l)["case_id"]: json.loads(l) for l in open(a.llm_proposals)} if a.llm_proposals else {}
    done = {json.loads(l)["case_id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    specs = [c for c in json.load(open(a.cases))["cases"] if c["split"] == a.split and c["case_id"] not in done]
    if a.llm_proposals:
        missing = [c["case_id"] for c in specs if c["case_id"] not in llm]
        if missing:
            sys.exit(f"LLM proposals missing for {len(missing)} cases, e.g. {missing[:3]}")
    specs = specs[: a.limit] if a.limit else specs
    jobs = [(c, a.config, a.climate, fm, llm.get(c["case_id"]) if a.llm_proposals else None) for c in specs]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_case, j): j[0]["case_id"] for j in jobs}
        for f in as_completed(futs):
            try:
                rows = f.result()
            except Exception:
                rows = [dict(case_id=futs[f], split=a.split, proposer="error", error=traceback.format_exc(limit=4))]
            with open(a.out, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            msg = " | ".join(f"{r['proposer']}: s={r.get('s') if r.get('s') is None else round(r['s'])} d={r.get('plant_delta') if r.get('plant_delta') is None else round(r['plant_delta'])}" for r in rows)
            print(futs[f], msg, flush=True)
