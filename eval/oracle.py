"""Oracle of EVAL_PROTOCOL.md: evaluate every candidate action list on every case (8 CRN replications, 12 h) and mark
the acceptable set. Parallel over (case, candidate). usage: oracle.py TWIN_V2_CONFIG CLIMATE_CSV SCENARIOS_JSON OUT_JSON [--workers N]"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HORIZON_H, REPS, acceptable_set, build_state, candidates, evaluate, load_inputs  # noqa: E402

_state = {}


def work(args):
    cfg_p, clim_p, spec, name, actions = args
    cfg, clim = load_inputs(cfg_p, clim_p)
    if spec["case_id"] not in _state:
        _state.clear()
        _state[spec["case_id"]] = build_state(spec, cfg, clim)
    r = evaluate(_state[spec["case_id"]], actions, HORIZON_H, REPS)
    return spec["case_id"], name, r["net"], r["mean"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("climate")
    ap.add_argument("scenarios")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    cfg, _ = load_inputs(a.config, a.climate)
    cases = json.load(open(a.scenarios))["cases"]
    cands = candidates(cfg)
    jobs = [(a.config, a.climate, c, n, acts) for c in cases for n, acts in cands]  # grouped by case: state cache hits
    t0 = time.time()
    res = {c["case_id"]: {} for c in cases}
    kp = {c["case_id"]: {} for c in cases}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (cid, name, net, mean) in enumerate(ex.map(work, jobs, chunksize=len(cands) // a.workers + 1)):
            res[cid][name], kp[cid][name] = net, mean
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    out = dict(protocol="EVAL_PROTOCOL.md", horizon_h=HORIZON_H, reps=REPS, candidates={n: acts for n, acts in cands}, cases={})
    for c in cases:
        best, table = acceptable_set(res[c["case_id"]])
        out["cases"][c["case_id"]] = dict(best=best, best_mean=table[best]["mean"], no_action_mean=table["no_action"]["mean"],
                                          n_acceptable=sum(v["acceptable"] for v in table.values()), table=table,
                                          kpis_best=kp[c["case_id"]][best], kpis_no_action=kp[c["case_id"]]["no_action"], net=res[c["case_id"]])
        o = out["cases"][c["case_id"]]
        print(c["case_id"], c["inject"], "best:", best, o["best_mean"], "no_action:", o["no_action_mean"], "acceptable:", o["n_acceptable"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"oracle done in {time.time() - t0:.0f}s")
