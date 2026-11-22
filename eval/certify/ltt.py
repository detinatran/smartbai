"""Learn-then-Test gate (CERTIFY_PROTOCOL.md, Method step 5, amendment 1 on screening, amendment 4 on the test order,
amendment 5 on one gate per family)."""
import math

import numpy as np
from scipy.stats import binom

ALPHA, DELTA = 0.10, 0.10
GRID = np.arange(3000.0, -1000.0 - 1e-9, -50.0)  # amendments 1-3 (v1, v2): most conservative first
GRID_A4 = np.arange(0.0, -1000.0 - 1e-9, -50.0)  # amendment 4: from tau = 0 (the per-case twin bound) downwards
GRIDS = dict(a2=GRID, a4=GRID_A4)


def n_min(alpha=ALPHA, delta=DELTA):
    """smallest number of executed calibration cases with which a threshold can be certified (zero harms)."""
    return math.ceil(math.log(delta) / math.log(1 - alpha))


def ltt_threshold(scores, harms, alpha=ALPHA, delta=DELTA, grid=GRID):
    """scores: s per calibration case (-inf when there is no candidate); harms: 0/1 harm of the chosen action on the plant.
    Fixed-sequence testing of H_k: P(harm | s > tau_k) > alpha with exact binomial p-values. Thresholds selecting fewer
    than n_min cases are screened out (neither tested nor a reason to stop). Returns (tau, trace); tau is None when no
    threshold is certified (the method then always abstains)."""
    s, h = np.asarray(scores, float), np.asarray(harms, float)
    tau, trace, m = None, [], n_min(alpha, delta)
    for t in grid:
        sel = s > t
        N, K = int(sel.sum()), int(h[sel].sum())
        if N < m:
            trace.append(dict(tau=float(t), N=N, K=K, p=None, status="screened"))
            continue
        p = float(binom.cdf(K, N, alpha))
        if p > delta:
            trace.append(dict(tau=float(t), N=N, K=K, p=p, status="stop"))
            break
        tau = float(t)
        trace.append(dict(tau=float(t), N=N, K=K, p=p, status="certified"))
    return tau, trace


def ltt_by_family(scores, harms, families, alpha=ALPHA, delta=DELTA, grid=GRID_A4):
    """amendment 5: one LTT threshold per family, each from that family's calibration cases only, at level delta per
    family. Returns ({family: tau}, {family: trace})."""
    s, h, f = np.asarray(scores, float), np.asarray(harms, float), np.asarray(families)
    taus, traces = {}, {}
    for fam in sorted(set(f.tolist())):
        m = f == fam
        taus[fam], traces[fam] = ltt_threshold(s[m], h[m], alpha, delta, grid)
    return taus, traces
