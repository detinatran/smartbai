"""Fit the twin's degradation model (CERTIFY_PROTOCOL.md amendment 1): P(station degraded now | sensors of the last hour),
logistic on the two features of failure_score, fitted on simulations with the nominal parameters (never on plant cases).
usage: fit_failure_model.py TWIN_V2_CONFIG CLIMATE_CSV OUT_JSON [--runs 24 --days 3]"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from line_sim import LineSim, load_climate  # noqa: E402
from twin_sync import failure_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("climate")
ap.add_argument("out")
ap.add_argument("--runs", type=int, default=24)
ap.add_argument("--days", type=float, default=3.0)
a = ap.parse_args()
cfg, climate = json.load(open(a.config)), load_climate(a.climate)
rng = np.random.default_rng(900000)
X, y = [], []
for k in range(a.runs):
    start = pd.Timestamp("2025-05-01 06:00") + pd.Timedelta(days=int(rng.integers(0, 150)), hours=int(rng.integers(0, 12)))
    sim = LineSim(cfg, climate, start=str(start), seed=900000 + k)
    sim.run(6 * 60)
    for _ in range(int(a.days * 24)):
        sim.run(60)
        for s in sim.ids:
            p, ev = failure_score(sim, s)
            if p is not None:
                X.append([ev["vibration_excess"], ev["motor_temp_excess_c"]])
                y.append(float(sim.S[s]["degraded"]))
    print(k, len(y), int(sum(y)), flush=True)
X, y = np.array(X), np.array(y)
Z = np.c_[np.ones(len(X)), X]
w = np.zeros(3)
for _ in range(50):  # Newton steps with a small ridge
    p = 1 / (1 + np.exp(-Z @ w))
    g = Z.T @ (p - y) + 1e-3 * w
    H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-3 * np.eye(3)
    step = np.linalg.solve(H, g)
    w -= step
    if np.abs(step).max() < 1e-8:
        break
p = 1 / (1 + np.exp(-Z @ w))
bins = np.digitize(p, [0.05, 0.1, 0.2, 0.4, 0.6])
calib = [(int(b), int((bins == b).sum()), round(float(p[bins == b].mean()), 3), round(float(y[bins == b].mean()), 3)) for b in np.unique(bins)]
json.dump(dict(intercept=w[0], vibration_excess=w[1], motor_temp_excess_c=w[2], n=int(len(y)), positives=int(y.sum()),
               calibration_bins=calib), open(a.out, "w"), indent=1)
print("coef", w.round(3), "base rate", round(y.mean(), 4), "calibration (bin, n, mean p, observed)", calib)
