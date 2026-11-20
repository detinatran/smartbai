"""smartbai: derive the production-line structure from the Bosch Production Line Performance data (Kaggle; the data
are subject to the competition rules and are NOT redistributed in this repo).
Reads train_date.csv.zip and train_numeric.csv.zip in chunks (no extraction) and writes small aggregate tables:
  stations.csv     line, station, #features, share of parts visiting, failure rate of visitors, time offsets
  transitions.csv  consecutive station pairs (ordered by timestamp) with counts and median transit time
  paths.csv        most frequent station sets (routes) with counts and failure rates
  summary.json     part counts, overall failure rate, time span
usage: bosch_line_structure.py DATA_DIR OUT_DIR [--max-rows N]"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--max-rows", type=int, default=None)
ap.add_argument("--chunk", type=int, default=50000)
a = ap.parse_args()
D, O = Path(a.data), Path(a.out)
O.mkdir(parents=True, exist_ok=True)
pat = re.compile(r"L(\d+)_S(\d+)_[DF](\d+)")

dcols = pd.read_csv(D / "train_date.csv.zip", nrows=0).columns.tolist()
ncols = pd.read_csv(D / "train_numeric.csv.zip", nrows=0).columns.tolist()
stations = sorted({(int(m[1]), int(m[2])) for c in dcols if (m := pat.match(c))}, key=lambda t: t[1])
sidx = {s: i for i, (_, s) in enumerate(stations)}
n_date = Counter(int(pat.match(c)[2]) for c in dcols if pat.match(c))
n_num = Counter(int(pat.match(c)[2]) for c in ncols if pat.match(c))
st_cols = {s: [c for c in dcols if (m := pat.match(c)) and int(m[2]) == s] for _, s in stations}
print(f"{len(stations)} stations on lines {sorted({l for l, _ in stations})}; {len(dcols) - 1} date / {len(ncols) - 2} numeric features")

# station time per part = earliest timestamp among that station's date features
T, ids = [], []
for k, ch in enumerate(pd.read_csv(D / "train_date.csv.zip", chunksize=a.chunk, dtype=np.float32, nrows=a.max_rows)):
    ids.append(ch["Id"].to_numpy(np.int64))
    T.append(np.stack([ch[st_cols[s]].min(axis=1).to_numpy(np.float32) for _, s in stations], axis=1))
    print(f"date chunk {k}: {sum(len(x) for x in ids)} parts", flush=True)
T, ids = np.concatenate(T), np.concatenate(ids)
y = pd.concat(pd.read_csv(D / "train_numeric.csv.zip", usecols=["Id", "Response"], chunksize=200000, nrows=a.max_rows))
y = y.set_index("Id").reindex(ids)["Response"].to_numpy()
vis = ~np.isnan(T)
start = np.nanmin(np.where(vis, T, np.nan), axis=1)

rows = []
for (l, s), i in zip(stations, range(len(stations))):
    v = vis[:, i]
    off = (T[v, i] - start[v]) if v.any() else np.array([np.nan])
    rows.append(dict(line=l, station=s, n_numeric=n_num.get(s, 0), n_date=n_date.get(s, 0), parts=int(v.sum()),
                     share_parts=float(v.mean()), fail_rate=float(y[v].mean()) if v.any() else np.nan,
                     median_offset=float(np.nanmedian(off)), p90_offset=float(np.nanpercentile(off, 90))))
pd.DataFrame(rows).to_csv(O / "stations.csv", index=False)

trans, dts, paths = Counter(), {}, Counter()
pfail = Counter()
names = [s for _, s in stations]
for r in range(len(T)):
    js = np.where(vis[r])[0]
    if not len(js):
        continue
    order = js[np.lexsort((js, T[r, js]))]          # by time, ties by station number
    key = "-".join(f"S{names[j]}" for j in sorted(js))
    paths[key] += 1
    pfail[key] += int(y[r] == 1)
    for p, q in zip(order[:-1], order[1:]):
        e = (names[p], names[q])
        trans[e] += 1
        if trans[e] <= 20000:
            dts.setdefault(e, []).append(T[r, q] - T[r, p])
tr = pd.DataFrame([dict(src=f"S{p}", dst=f"S{q}", count=c, median_dt=float(np.median(dts[(p, q)])))
                   for (p, q), c in trans.most_common()])
tr.to_csv(O / "transitions.csv", index=False)
pa = pd.DataFrame([dict(route=k, count=c, fail_rate=pfail[k] / c) for k, c in paths.most_common(200)])
pa.to_csv(O / "paths.csv", index=False)
summ = dict(parts=int(len(T)), fail_rate=float(np.nanmean(y)), failures=int(np.nansum(y)),
            stations=len(stations), lines=sorted({l for l, _ in stations}), distinct_routes=len(paths),
            time_min=float(np.nanmin(T)), time_max=float(np.nanmax(T)),
            median_stations_per_part=float(np.median(vis.sum(1))))
json.dump(summ, open(O / "summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
