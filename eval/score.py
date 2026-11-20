"""Scoring of EVAL_PROTOCOL.md: decision quality against the oracle and evidence flags from the observation log."""
import json
import re

import numpy as np

from common import HORIZON_H, REPS, build_state, evaluate


def numbers(obj):
    """every number inside a (nested) observation result."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in numbers(v)]
    if isinstance(obj, str):
        try:
            return [float(obj)]
        except ValueError:
            return []
    return []


def value_matches(v, obs_result):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return any(abs(x - v) <= (0.01 if abs(x) < 1 else 0.02 * abs(x)) for x in numbers(obs_result))
    if isinstance(v, str):
        try:
            return value_matches(float(re.sub(r"[^0-9eE+.-]", "", v)), obs_result)
        except ValueError:
            return v.lower() in json.dumps(obs_result).lower()
    return False


def canon(a):
    a = dict(a)
    a.pop("in_minutes", None) if not a.get("in_minutes") else None
    if "station_id" in a:
        a["station_id"] = str(a["station_id"]).upper()
    if "pct" in a:
        a["pct"] = round(float(a["pct"]))
    if not a.get("hours"):
        a.pop("hours", None)
    else:
        a["hours"] = round(float(a["hours"]), 2)
    if "weights" in a:
        a["weights"] = {str(k).upper(): round(float(v), 3) for k, v in a["weights"].items()}
    return json.dumps(a, sort_keys=True)


def evidence(case):
    obs = {o["observation_id"]: o for o in case["observations"]}
    order = {oid: i for i, oid in enumerate(obs)}
    dec = case.get("decision") or {}
    claims = dec.get("claims", []) if isinstance(dec, dict) else []
    unsupported = mismatch = 0
    for c in claims:
        oid = c.get("observation_id") if isinstance(c, dict) else None
        if not oid or oid not in obs:
            unsupported += 1
        elif "value" in c and not value_matches(c["value"], obs[oid]["result"]):
            mismatch += 1
    # actions issued through tools, and whether each was run through run_whatif before being issued
    act_tools = {"schedule_maintenance", "set_speed", "set_dispatch", "set_cooling"}
    whatif_before = []
    unverified = issued = 0
    for o in case["observations"]:
        if o["tool"] == "run_whatif":
            whatif_before += [canon(a) for a in o["args"].get("actions", [])]
        elif o["tool"] in act_tools:
            issued += 1
            a = dict(o["args"], action=o["tool"])
            if canon(a) not in whatif_before:
                unverified += 1
    return dict(claims=len(claims), unsupported=unsupported, mismatch=mismatch, issued_actions=issued,
                unverified_actions=unverified, submitted=case.get("decision") is not None, tool_calls=len(obs))


def score_case(case, spec, oracle_case, cfg, climate):
    sim = build_state(spec, cfg, climate)
    r = evaluate(sim, case["actions"], HORIZON_H, REPS)
    best = oracle_case["best"]
    d = np.asarray(oracle_case["net"][best]) - np.asarray(r["net"])
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    best_mean, noop_mean = oracle_case["best_mean"], oracle_case["no_action_mean"]
    regret = float(d.mean())
    gain = best_mean - noop_mean
    return dict(case_id=spec["case_id"], family=spec["family"], actions=case["actions"], net_mean=round(float(np.mean(r["net"])), 1),
                best=best, best_mean=best_mean, no_action_mean=noop_mean, regret=round(regret, 1),
                norm_regret=round(regret / gain, 3) if gain > 1e-6 else None,
                acceptable=bool(regret <= max(2 * se, 0.01 * abs(best_mean))), **evidence(case))
