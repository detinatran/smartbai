"""Summarise scored runs (EVAL_PROTOCOL.md primary comparison): acceptable rate, regret and evidence flags per condition
and family; paired Wilcoxon on regret between conditions (exploratory, two-sided).
usage: summarize.py RESULTS_1.jsonl [RESULTS_2.jsonl ...] [--oracle ORACLE_JSON]"""
import argparse
import json
from itertools import combinations

import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--oracle")
a = ap.parse_args()
rows = []
for f in a.results:
    for line in open(f):
        r = json.loads(line)
        r["run"] = f"{r['agent']}/{r['condition']}"
        rows.append(r)
d = pd.DataFrame(rows)
d["has_error"] = d["error"].notna()
agg = dict(cases=("case_id", "count"), acceptable=("acceptable", "mean"), regret=("regret", "mean"), norm_regret=("norm_regret", "mean"),
           unsupported=("unsupported", "sum"), mismatch=("mismatch", "sum"), claims=("claims", "sum"),
           unverified=("unverified_actions", "sum"), issued=("issued_actions", "sum"), tool_calls=("tool_calls", "mean"), errors=("has_error", "sum"))
print("== per run")
print(d.groupby("run").agg(**agg).round(3).to_string())
print("\n== per run x family (acceptable rate / mean regret)")
print(d.pivot_table(index="family", columns="run", values=["acceptable", "regret"], aggfunc="mean").round(3).to_string())
if a.oracle:
    o = json.load(open(a.oracle))["cases"]
    print("\n== oracle: best candidate, gain over no action, size of the acceptable set")
    for cid, c in o.items():
        print(f"{cid:15s} best={c['best']:22s} gain={c['best_mean'] - c['no_action_mean']:9.1f} acceptable={c['n_acceptable']}")
runs = sorted(d["run"].unique())
if len(runs) > 1:
    print("\n== paired Wilcoxon on regret (two-sided, exploratory)")
    p = d.pivot_table(index="case_id", columns="run", values="regret")
    for x, y in combinations(runs, 2):
        q = p[[x, y]].dropna()
        diff = q[x] - q[y]
        pv = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        print(f"{x} - {y}: mean diff {diff.mean():.1f} over {len(q)} cases, p = {pv:.4f}")
