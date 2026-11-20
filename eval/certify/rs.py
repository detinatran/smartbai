"""Sequential paired comparison against no action, in the twin (CERTIFY_PROTOCOL.md, Method step 2-3)."""
import math

import numpy as np
from scipy.stats import t as student_t

K_MAX, N0, STEP, BUDGET, CONF = 5, 4, 2, 64, 0.95


def certify_rs(twin, candidates, hours=12.0, budget=BUDGET, n0=N0, step=STEP, conf=CONF):
    """twin: TwinModel; candidates: list of action lists (no-action excluded, at most K_MAX).
    Returns the chosen candidate, its score s (lower bound of the mean paired gain over no action) and the trace."""
    cands = [c for c in candidates if c][:K_MAX]
    if not cands:
        return dict(j_star=None, s=-math.inf, used=0, n=0, alive=[], means={}, ses={})
    net = {}

    def run(j, r):
        if (j, r) not in net:
            net[(j, r)] = twin.run(cands[j - 1] if j else [], r, hours)["net_value"]
        return net[(j, r)]

    alive = list(range(1, len(cands) + 1))
    n = n0
    for r in range(n):
        for j in [0] + alive:
            run(j, r)
    used = n * (len(alive) + 1)

    def stats(j):
        d = np.array([net[(j, r)] - net[(0, r)] for r in range(n)])
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))

    while True:
        tq = float(student_t.ppf(conf, n - 1))
        st = {j: stats(j) for j in alive}
        best_lcb = max(m - tq * se for m, se in st.values())
        alive = [j for j in alive if st[j][0] + tq * st[j][1] >= best_lcb]
        cost = step * (len(alive) + 1)
        if len(alive) <= 1 or used + cost > budget:
            break
        for r in range(n, n + step):
            for j in [0] + alive:
                run(j, r)
        n += step
        used += cost
    tq = float(student_t.ppf(conf, n - 1))
    st = {j: stats(j) for j in alive}
    j_star = max(alive, key=lambda j: st[j][0])
    m, se = st[j_star]
    return dict(j_star=j_star - 1, actions=cands[j_star - 1], s=m - tq * se, mean_gain=m, se=se, used=used, n=n,
                alive=[j - 1 for j in alive], means={j - 1: v[0] for j, v in st.items()})
