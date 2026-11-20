"""Proposers of CERTIFY_PROTOCOL.md. A proposer returns up to K = 5 candidate action lists for a case.

- rule: the fixed-rule agent of eval/agents.py, run in-process on the case; its full action list, then each single action.
- random5: 5 candidates drawn from the 163 non-null candidates of eval/common.py, seeded per case.
- llm: an LLM agent through the HTTP tool server with read tools only. It ends with
      submit_decision(report={"candidates": [[action, ...], ...], "rationale": ..., "claims": [...]})
  where an action is {"action": "schedule_maintenance", "station_id": "S5", "in_minutes": 0} |
  {"action": "set_speed", "station_id": ..., "pct": ...} | {"action": "set_dispatch", "weights": {...}, "hours": ...} |
  {"action": "set_cooling", "level": "high"}.

CLI (LLM proposals, written to a JSONL file that run.py reads):
  proposers.py --server http://127.0.0.1:8765 --agent mymodule:my_agent --cases CASES_JSON --split dev --out PROPOSALS_JSONL
"""
import argparse
import importlib
import json
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim"))
from agents import ToolClient, rule_agent  # noqa: E402
from common import candidates as all_candidates  # noqa: E402
from tool_server import TOOLS, Twin  # noqa: E402

ACTION_TOOLS = ["schedule_maintenance", "set_speed", "set_dispatch", "set_cooling"]
READ_ONLY = ACTION_TOOLS + ["run_whatif"]  # tools disabled for every proposer
K = 5


class LocalClient:
    """the ToolClient interface on an in-process tool-server twin (no HTTP)."""

    def __init__(self, twin):
        self.twin = twin

    def call(self, tool, **args):
        if tool not in TOOLS:
            return dict(error=f"unknown tool {tool}", http_status=404)
        if tool in self.twin.disabled:
            return dict(error=f"tool {tool} is not available in this case", http_status=403)
        try:
            return self.twin.observe(tool, args, getattr(self.twin, tool)(args))
        except (ValueError, KeyError, TypeError) as e:
            return dict(error=f"{type(e).__name__}: {e}", http_status=400)

    def task(self):
        return dict(task=self.twin.spec.get("task", ""), tools=[t for t in TOOLS if t not in self.twin.disabled])


def dedupe(lists, k=K):
    out, seen = [], set()
    for c in lists:
        key = json.dumps(c, sort_keys=True)
        if c and key not in seen:
            seen.add(key)
            out.append(c)
    return out[:k]


def rule_proposals(spec, cfg, climate):
    """returns (candidates, full rule action list). The rule agent's action tools act on a scratch copy of the case."""
    tw = Twin(cfg, climate)
    tw.reset(dict(spec, disabled_tools=["run_whatif"]))
    rule_agent(LocalClient(tw), spec.get("task", ""))
    acts = list(tw.actions)
    return dedupe([acts] + [[a] for a in acts]), acts


def random_proposals(spec, cfg, k=K):
    cands = [c for name, c in all_candidates(cfg) if name != "no_action"]
    rng = np.random.default_rng([int(spec["seed"]), 5])
    return [cands[i] for i in rng.choice(len(cands), size=k, replace=False)]


def valid_action(a, station_ids):
    try:
        kind = a["action"]
        if kind == "schedule_maintenance":
            return a["station_id"] in station_ids and float(a.get("in_minutes", 0)) >= 0
        if kind == "set_speed":
            return a["station_id"] in station_ids and 30 <= float(a["pct"]) <= 150
        if kind == "set_dispatch":
            return all(s in station_ids for s in a["weights"]) and (a.get("hours") is None or float(a["hours"]) > 0)
        if kind == "set_cooling":
            return a.get("level") in ("normal", "high")
    except (KeyError, TypeError, ValueError):
        return False
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--agent", required=True, help="module:function, called as agent(client, task)")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "calibration", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mod, fn = a.agent.split(":")
    agent = getattr(importlib.import_module(mod), fn)
    client = ToolClient(a.server)
    done = {json.loads(l)["case_id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    for spec in json.load(open(a.cases))["cases"]:
        if spec["split"] != a.split or spec["case_id"] in done:
            continue
        client.admin("reset", dict(spec, disabled_tools=READ_ONLY))
        err = None
        try:
            agent(client, client.task()["task"])
        except Exception:
            err = traceback.format_exc(limit=3)
        case = client.admin("case")
        rep = case.get("decision") or {}
        sids = {f"S{i}" for i in range(39)}  # station ids of twin v2
        raw = rep.get("candidates", []) if isinstance(rep, dict) else []
        cands = [[x for x in c if valid_action(x, sids)] for c in raw if isinstance(c, list)]
        row = dict(case_id=spec["case_id"], proposer="llm", candidates=dedupe(cands), n_raw=len(raw),
                   n_invalid_actions=sum(len(c) for c in raw if isinstance(c, list)) - sum(len(c) for c in cands),
                   tool_calls=len(case.get("observations", [])), report=rep, error=err)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(spec["case_id"], len(row["candidates"]), "candidates", "err" if err else "", flush=True)
