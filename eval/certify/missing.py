"""Print the number of cases of a split that have no result row yet (error rows do not count as results).
usage: missing.py CASES_JSON SPLIT OUT_JSONL"""
import json
import os
import sys

cases, split, out = sys.argv[1:4]
ids = [c["case_id"] for c in json.load(open(cases))["cases"] if c["split"] == split]
done = {r["case_id"] for r in map(json.loads, open(out)) if r["proposer"] != "error"} if os.path.exists(out) else set()
print(sum(k not in done for k in ids))
