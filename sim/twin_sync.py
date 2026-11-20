"""Digital twin synchronised from plant observables only (CERTIFY_PROTOCOL.md, section "Plant and twin").

The plant is a LineSim with its own (hidden) parameters and hidden state. The twin copies what an operator can observe
(station up/down, queues, speeds, dispatch, cooling, scheduled maintenance, parts in progress, the technician's repair
estimate, the temperature forecast, the rush target) and replaces everything else:
- physical parameters: nominal values (ASSUMED unless overridden);
- per-station defect probabilities: the nominal calibration;
- degradation: sampled per replication, station i degraded with probability p_i from the fitted degradation model
  (fit_failure_model.py, amendment 1);
- defect drift: estimated from recent completed parts (Poisson test with greedy attribution, amendment 1);
- failure clocks: fresh.
Replication r samples its hidden state from the stream [seed, r] and runs on [seed, r, 1], so every candidate action list
asked at the same twin seeds the same states and streams (common random numbers).
"""
import copy
import heapq
import math
from functools import lru_cache

import numpy as np

from line_sim import ASSUMED, LineSim


def failure_score(sim, sid, horizon_h=4.0):
    """probability of failure within horizon_h from load-adjusted vibration and motor temperature of the last hour
    (the `predict_failure` baseline of the tool server). None when the station is not running."""
    S = sim.S[sid]
    vib = [v for _, v in sim.sensor_history(sid, "vibration", 1.0)]
    tmp = [v for _, v in sim.sensor_history(sid, "motor_temp", 1.0)]
    if not vib or S["state"] != "up":
        return None, None
    util = min(1.0, S["busy_time"] / max(sim.t, 1.0))
    excess_v = float(np.mean(vib)) - (0.9 + 0.6 * util)
    excess_t = float(np.mean(tmp)) - (sim.hall_c() + 14 * util)
    z = 4.0 * excess_v + 0.25 * excess_t - 1.5
    p_now = 1 / (1 + math.exp(-z))
    p = 1 - (1 - p_now) ** max(horizon_h / 4.0, 0.25)
    return p, dict(vibration_excess=round(excess_v, 3), motor_temp_excess_c=round(excess_t, 2))


_NOMINAL = {}


def nominal_defect(sim, params):
    """per-station defect probability of a LineSim built with the nominal parameters on the same line."""
    key = (id(sim.cfg), tuple(sorted((params or {}).items())))
    if key not in _NOMINAL:
        ref = LineSim(sim.cfg, sim.climate, start=str(sim.t0), seed=0, params=params)
        _NOMINAL[key] = {s: ref.S[s]["p_defect"] for s in ref.ids}
    return _NOMINAL[key]


def estimate_drift(sim, p_nom, hours=4.0, alpha=0.01, max_iter=3):
    """defect multipliers from parts completed in the last `hours` (amendment 1 of CERTIFY_PROTOCOL.md).
    For station i: n_i parts visited it, k_i of them were defective, E_i is their expected number of defects under the
    current estimates. The station with the smallest Poisson tail p-value P(Poisson(E_i) >= k_i), if below alpha / 39
    (Bonferroni), gets the excess attributed to it: drift_i = 1 + (k_i - E_i) / (sum over those parts of p_i).
    Repeated up to max_iter times, each time with the expectations updated."""
    from scipy.stats import poisson
    lo = sim.t - hours * 60.0
    done = [c for c in sim.completed if c[0] >= lo]
    drift = {s: 1.0 for s in sim.ids}
    if not done:
        return drift
    paths = [set(c[4]) for c in done]
    bad = np.array([c[2] for c in done], float)
    visit = {s: np.array([s in pth for pth in paths]) for s in sim.ids}
    for _ in range(max_iter):
        p_part = np.array([1 - np.prod([1 - min(p_nom[s] * drift[s], 1.0) for s in pth]) for pth in paths])
        best, best_p = None, alpha / len(sim.ids)
        for s in sim.ids:
            m = visit[s]
            if m.sum() < 30 or drift[s] != 1.0:
                continue
            k, e = bad[m].sum(), p_part[m].sum()
            pv = float(poisson.sf(k - 1, e)) if k > e else 1.0
            if pv < best_p:
                best, best_p = (s, m, k, e), pv
        if best is None:
            break
        s, m, k, e = best
        drift[s] = 1.0 + (k - e) / max(m.sum() * p_nom[s], 1e-9)
    return drift


def degraded_prob(sim, sid, model):
    """P(station degraded now) from the fitted degradation model (fit_failure_model.py); 0 when it is not running."""
    p, ev = failure_score(sim, sid)
    if p is None:
        return 0.0
    z = model["intercept"] + model["vibration_excess"] * ev["vibration_excess"] + model["motor_temp_excess_c"] * ev["motor_temp_excess_c"]
    return float(1 / (1 + math.exp(-z)))


class TwinModel:
    def __init__(self, plant, seed, failure_model, params=None, forecast_bias=0.0, horizon_h=12.0):
        self.seed = [int(x) for x in (seed if isinstance(seed, (list, tuple)) else [seed])]
        P = dict(ASSUMED, **(params or {}))
        tw = copy.deepcopy(plant)
        tw.P = dict(P)
        tw.base_coupling = P["hall_coupling"]
        tw.P["hall_coupling"] = P["cooling_high_coupling"] if plant.cooling == "high" else P["hall_coupling"]
        p_nom = nominal_defect(plant, params)
        drift = estimate_drift(plant, p_nom)
        self.p_fail = {}
        for s in tw.ids:
            S = tw.S[s]
            S["p_defect"], S["drift"], S["degraded"], S["haz"] = p_nom[s], drift[s], False, 0.0
            if S["state"] == "up":
                self.p_fail[s] = degraded_prob(plant, s, failure_model)
            else:  # stopped: the twin only knows the technician's estimate of when it is back
                S["down_until"] = max(S["est_until"], tw.t + 1.0)
        # restart events at the true repair times are replaced by events at the estimated times
        tw.ev = [e for e in tw.ev if e[2] != "restart"]
        heapq.heapify(tw.ev)
        for s in tw.ids:
            S = tw.S[s]
            if S["state"] != "up":
                tw.push(S["down_until"], "restart", (s, S["state"], S["down_until"] - S["stop_t"]))
        # temperature over the horizon: the forecast (plant temperature, including an announced heat wave, plus a bias)
        delta = plant.heat[2] if plant.heat and plant.heat[0] <= plant.t < plant.heat[1] else 0.0
        tw.heat = (tw.t, tw.t + 60.0 * horizon_h, delta + float(forecast_bias)) if (delta or forecast_bias) else None
        self.base, self.P = tw, P

    def replica(self, r):
        sim = copy.deepcopy(self.base)
        rng = np.random.default_rng(self.seed + [int(r)])
        win = self.P["precursor_h"] / self.P["mtbf_h"]
        for s in sim.ids:
            S = sim.S[s]
            S["haz_E"] = float(rng.exponential())
            if s in self.p_fail and rng.random() < self.p_fail[s]:
                S["degraded"], S["haz"] = True, max(0.0, S["haz_E"] - float(rng.uniform()) * win)
        sim.rng = np.random.default_rng(self.seed + [int(r), 1])
        return sim

    def run(self, actions, r, hours):
        sim = self.replica(r)
        t0, base = sim.t, sim.counters()
        sim.apply(actions)
        sim.run(hours * 60.0)
        return sim.kpis(since=t0, base=base)

    def whatif(self, actions, hours, reps=5, first_rep=0):
        runs = [self.run(actions, first_rep + r, hours) for r in range(reps)]
        keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
        return dict(hours=hours, reps=reps, actions=actions,
                    mean={k: round(float(np.mean([x[k] for x in runs])), 5) for k in keys},
                    sd={k: round(float(np.std([x[k] for x in runs], ddof=1)), 5) for k in keys} if reps > 1 else None,
                    runs=runs)
