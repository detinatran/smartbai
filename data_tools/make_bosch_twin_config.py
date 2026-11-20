"""smartbai twin v2: build the line configuration from the Bosch structure tables (bosch_line_structure.py output).
The stage/lane layout below was read off the transition table (stations ordered by timestamp); every number attached to
a station or edge (share of parts, failure rate, #sensors, time offset, flow counts) comes from the data.
The output JSON is derived from Kaggle competition data, so it stays out of the public repo.
usage: make_bosch_twin_config.py STRUCTURE_DIR OUT_JSON"""
import json
import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1])
st = pd.read_csv(S / "stations.csv").set_index("station")
tr = pd.read_csv(S / "transitions.csv")
summ = json.load(open(S / "summary.json"))

# lanes: list of stages, each stage = parallel stations. Merge point: every lane feeds S29 (trunk).
LANES = {
    "L0A": dict(line=0, stages=[[0], [1], [2, 3], [4, 5], [6, 7], [8], [9, 10, 11]]),
    "L0B": dict(line=0, stages=[[12], [13], [14, 15], [16, 17], [18, 19], [20], [21, 22, 23]]),
    "L1-L2": dict(line=1, stages=[[24, 25], [26, 27, 28]]),
    "L3": dict(line=3, stages=[[29], [30], [31, 32], [33], [34], [35, 36], [37], [38]]),
}
OPTIONAL = {31, 32, 38}  # side / re-test stations visited by a minority of parts

stations, edges = [], []
for lane, d in LANES.items():
    for k, grp in enumerate(d["stages"]):
        for j, s in enumerate(grp):
            r = st.loc[s]
            stations.append(dict(
                id=f"S{s}", line=int(r.line), lane=lane, stage=k, slot=j, parallel_group=[f"S{x}" for x in grp],
                optional=s in OPTIONAL, share_parts=round(float(r.share_parts), 5), fail_rate=round(float(r.fail_rate), 5),
                parts=int(r.parts), n_sensors=int(r.n_numeric), median_offset=round(float(r.median_offset), 3)))
ids = {x["id"] for x in stations}
for _, e in tr.iterrows():
    if e.src in ids and e.dst in ids and e["count"] >= 1000:
        edges.append(dict(src=e.src, dst=e.dst, parts=int(e["count"]), median_dt=round(float(e.median_dt), 3)))
cfg = dict(name="smartbai_twin_v2_bosch", source="Bosch Production Line Performance (Kaggle), train set; aggregates only",
           parts=summ["parts"], fail_rate=summ["fail_rate"], time_unit="anonymised Bosch time unit",
           lanes={k: dict(line=v["line"], stages=[[f"S{s}" for s in g] for g in v["stages"]]) for k, v in LANES.items()},
           merge_into="S29", stations=stations, edges=edges,
           excluded="L3 sub-line S39-S51 (separate product family, 5% of parts) and S42/S46 (<20 parts)")
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
print(f"{len(stations)} stations, {len(edges)} edges, hot spot:",
      max(stations, key=lambda x: x["fail_rate"] if x["parts"] > 5000 else 0)["id"])
