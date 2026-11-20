"""Learn-then-Test gate (CERTIFY_PROTOCOL.md, Method step 5, and amendment 1 on screening)."""
import math

import numpy as np
from scipy.stats import binom

ALPHA, DELTA = 0.10, 0.10
GRID = np.arange(3000.0, -1000.0 - 1e-9, -50.0)  # most conservative first


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
