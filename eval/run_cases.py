"""Run an agent over the evaluation cases through the tool server and score each case (EVAL_PROTOCOL.md).
usage: run_cases.py --server http://127.0.0.1:8765 --agent rule|noop|module:function --condition a|b|c
                    --config TWIN_V2_CONFIG --climate CLIMATE_CSV --scenarios SCENARIOS_JSON --oracle ORACLE_JSON --out RESULTS_JSONL"""
import argparse
import importlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import AGENTS, ToolClient  # noqa: E402
from common import load_inputs  # noqa: E402
from score import score_case  # noqa: E402

DISABLED = dict(a=[], b=["run_whatif"], c=["run_whatif"])

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8765")
ap.add_argument("--agent", default="rule")
ap.add_argument("--condition", choices=list(DISABLED), default="c")
ap.add_argument("--config", required=True)
ap.add_argument("--climate", required=True)
ap.add_argument("--scenarios", required=True)
ap.add_argument("--oracle", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cases", nargs="*")
a = ap.parse_args()

if a.agent in AGENTS:
    agent = AGENTS[a.agent]
else:
    mod, fn = a.agent.split(":")
    agent = getattr(importlib.import_module(mod), fn)
cfg, climate = load_inputs(a.config, a.climate)
cases = json.load(open(a.scenarios))["cases"]
oracle = json.load(open(a.oracle))["cases"]
client = ToolClient(a.server)
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["case_id"] for l in open(a.out)}
for spec in cases:
    if (a.cases and spec["case_id"] not in a.cases) or spec["case_id"] in done:
        continue
    t0 = time.time()
    client.admin("reset", dict(spec, disabled_tools=DISABLED[a.condition]))
    err = None
    try:
        agent(client, client.task()["task"])
    except Exception:  # an agent crash is a recorded failure of the case, not a reason to stop
        err = traceback.format_exc(limit=3)
    case = client.admin("case")
    row = score_case(case, spec, oracle[spec["case_id"]], cfg, climate)
    row.update(agent=a.agent, condition=a.condition, error=err, sec=round(time.time() - t0, 1))
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row["case_id"], "acceptable" if row["acceptable"] else "NOT acceptable", "regret", row["regret"], "actions", row["actions"],
          "unsupported", row["unsupported"], "mismatch", row["mismatch"], "err" if err else "", flush=True)
