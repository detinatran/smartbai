"""Calibrate the gate on the calibration split and report the test split (CERTIFY_PROTOCOL.md: conditions, H1-H4, secondary).
usage: summarize.py --calibration CAL.jsonl [CAL_LLM.jsonl ...] --test TEST.jsonl [TEST_LLM.jsonl ...]
       summarize.py --dev DEV.jsonl     (descriptive only: tau chosen on the dev split itself)
       --ltt a2 reproduces v1 and v2 (amendments 1-3); the default a4 is the test order of amendment 4."""
import argparse
import json
import math

import numpy as np
from scipy.stats import beta, binomtest, wilcoxon

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltt import ALPHA, DELTA, GRIDS, ltt_threshold  # noqa: E402


def load(files):
    rows = {}
    for f in files or []:
        for line in open(f):
            r = json.loads(line)
            rows[(r["case_id"], r["proposer"])] = r  # later files override (e.g. a run with LLM proposals)
    return list(rows.values())


def by_proposer(rows, p):
    return sorted([r for r in rows if r["proposer"] == p], key=lambda r: r["case_id"])


def gate(rows, tau):
    """per case: executed?, plant gain G (0 when abstaining), harm (only when executed)."""
    out = []
    for r in rows:
        ex = r.get("j_star") is not None and tau is not None and r["s"] > tau
        out.append(dict(case_id=r["case_id"], family=r["family"], executed=ex, G=r["plant_delta"] if ex else 0.0,
                        harm=r.get("harm", 0) if ex else None))
    return out


def cp(k, n, level=0.95):
    if n == 0:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    return (0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k)))


def boot_mean(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def describe(name, g):
    ex = [x for x in g if x["executed"]]
    k = sum(x["harm"] for x in ex)
    lo, hi = cp(k, len(ex))
    m, blo, bhi = boot_mean([x["G"] for x in g])
    return dict(condition=name, cases=len(g), executed=len(ex), coverage=round(len(ex) / max(len(g), 1), 3), harms=k,
                harm_rate=round(k / len(ex), 3) if ex else None, harm_ci=(round(lo, 3), round(hi, 3)),
                mean_G=round(m, 1), G_ci=(round(blo, 1), round(bhi, 1)))


def conditions(rows, taus):
    out = {"no_action": [dict(case_id=r["case_id"], family=r["family"], executed=False, G=0.0, harm=None) for r in by_proposer(rows, "rules_direct")]}
    out["rules"] = [dict(case_id=r["case_id"], family=r["family"], executed=bool(r["actions"]), G=r["plant_delta"],
                         harm=r["harm"] if r["actions"] else None) for r in by_proposer(rows, "rules_direct")]
    for p in ("llm", "rule", "random5"):
        rs = by_proposer(rows, p)
        if not rs:
            continue
        out[f"{p}-ungated"] = gate(rs, -math.inf)
        out[f"{p}-naive"] = gate(rs, 0.0)
        out[f"{p}-certify"] = gate(rs, taus.get(p))
    return out


def risk_coverage(rs, grid=(1500, 1000, 500, 250, 0, -250, -500, -1000, -math.inf)):
    lines = []
    for t in grid:
        g = gate(rs, t)
        ex = [x for x in g if x["executed"]]
        lines.append(f"tau={t:>6}: coverage {len(ex) / len(g):.2f}, harm {sum(x['harm'] for x in ex) / max(len(ex), 1):.2f}, mean G {np.mean([x['G'] for x in g]):.0f}")
    return lines


def holm(pvals):
    order = sorted(pvals, key=lambda k: pvals[k])
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", nargs="*")
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--dev", nargs="*")
    ap.add_argument("--ltt", choices=sorted(GRIDS), default="a4", help="test order of the LTT grid: a2 (v1, v2) or a4 (amendment 4)")
    a = ap.parse_args()
    cal = load(a.dev or a.calibration)
    test = load(a.dev or a.test)
    label = "DEV (descriptive; tau chosen on the same cases)" if a.dev else "TEST"
    taus = {}
    print(f"== calibration (alpha={ALPHA}, delta={DELTA}, LTT order {a.ltt}: {GRIDS[a.ltt][0]:g} to {GRIDS[a.ltt][-1]:g})")
    for p in ("llm", "rule", "random5"):
        rs = by_proposer(cal, p)
        if rs:
            s = [r["s"] if r.get("j_star") is not None else -math.inf for r in rs]
            h = [r.get("harm", 0) for r in rs]
            taus[p], trace = ltt_threshold(s, h, grid=GRIDS[a.ltt])
            last = [t for t in trace if t["status"] != "screened"][-1:] if trace else []
            print(f"{p:8s} tau={taus[p]}  cases={len(rs)}  harm rate if all executed={np.mean([x for x, r in zip(h, rs) if r.get('j_star') is not None]):.3f}  last test={last}")
    cond = conditions(test, taus)
    print(f"\n== {label}: conditions")
    for name, g in cond.items():
        print(describe(name, g))
    main_p = "llm" if f"llm-certify" in cond else "rule"
    print(f"\n== hypotheses ({'LLM' if main_p == 'llm' else 'Rule-Certify in place of LLM (no LLM proposals)'})")
    cert, naive = cond[f"{main_p}-certify"], cond[f"{main_p}-naive"]
    ex_c, ex_n = [x for x in cert if x["executed"]], [x for x in naive if x["executed"]]
    k_c, k_n = sum(x["harm"] for x in ex_c), sum(x["harm"] for x in ex_n)
    p1 = binomtest(k_c, len(ex_c), ALPHA, alternative="greater").pvalue if ex_c else math.nan
    print(f"H1 validity: harms {k_c}/{len(ex_c)}, CP 95% upper {cp(k_c, len(ex_c))[1]:.3f}, p(R>alpha) = {p1:.4f} -> {'REFUTED' if p1 < 0.05 else 'not refuted'}")
    pv = {}
    pv["H2"] = binomtest(k_n, len(ex_n), ALPHA, alternative="greater").pvalue if ex_n else 1.0
    G = np.array([x["G"] for x in cert])
    pv["H3"] = wilcoxon(G, zero_method="wilcox", alternative="greater").pvalue if (G != 0).any() else 1.0
    if main_p == "llm" and "random5-certify" in cond:
        d = G - np.array([x["G"] for x in cond["random5-certify"]])
        pv["H4"] = wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue if (d != 0).any() else 1.0
    adj = holm(pv)
    print(f"H2 naive harm {k_n}/{len(ex_n)} = {k_n / max(len(ex_n), 1):.3f}")
    print(f"H3 certify mean G {G.mean():.1f}")
    for k in pv:
        print(f"{k}: p = {pv[k]:.4g}, Holm-adjusted = {adj[k]:.4g} -> {'supported' if adj[k] < 0.05 else 'not supported'}")
    print("\n== risk-coverage on the test cases (secondary)")
    for p in ("llm", "rule", "random5"):
        rs = by_proposer(test, p)
        if rs:
            print(p)
            for line in risk_coverage(rs):
                print("  " + line)
    print("\n== per family (coverage / harm rate / mean G)")
    fams = sorted({x["family"] for x in cert})
    for name in [n for n in cond if n.endswith("certify") or n in ("rules", "no_action")]:
        cells = []
        for f in fams:
            g = [x for x in cond[name] if x["family"] == f]
            ex = [x for x in g if x["executed"]]
            cells.append(f"{f}: {len(ex)}/{len(g)}, {sum(x['harm'] for x in ex)}, {np.mean([x['G'] for x in g]):.0f}")
        print(f"{name:16s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
