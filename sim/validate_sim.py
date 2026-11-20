"""smartbai: check that the simulator reproduces the Bosch line it was built from, and show the Dubai climate effect.
usage: validate_sim.py TWIN_V2_CONFIG CLIMATE_CSV OUT_JSON"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from line_sim import ASSUMED, LineSim, load_climate

cfg, clim, out = json.load(open(sys.argv[1])), load_climate(sys.argv[2]), sys.argv[3]
bosch = {s["id"]: s for s in cfg["stations"]}
res = dict(assumed=ASSUMED, runs=[])
for season, start in (("july", "2025-07-01 00:00"), ("january", "2025-01-06 00:00")):
    for seed in (0, 1, 2):
        t0 = time.time()
        sim = LineSim(cfg, clim, start=start, seed=seed).run(7 * 24 * 60)
        k = sim.kpis()
        done = sim.completed
        n = len(done)
        visits = {s: 0 for s in bosch}
        fails = {s: 0 for s in bosch}
        for c in done:
            for s in set(c[4]):
                visits[s] += 1
                fails[s] += c[2]
        share_sim = np.array([visits[s] / n for s in bosch])
        share_b = np.array([bosch[s]["share_parts"] for s in bosch])
        fr_sim = np.array([fails[s] / visits[s] if visits[s] > 200 else np.nan for s in bosch])
        fr_b = np.array([bosch[s]["fail_rate"] for s in bosch])
        m = ~np.isnan(fr_sim)
        res["runs"].append(dict(
            season=season, seed=seed, sec=round(time.time() - t0, 1), **k,
            share_corr=round(float(np.corrcoef(share_sim, share_b)[0, 1]), 4),
            share_mae=round(float(np.mean(np.abs(share_sim - share_b))), 4),
            share_max_abs=round(float(np.max(np.abs(share_sim - share_b))), 4),
            failrate_corr=round(float(np.corrcoef(fr_sim[m], fr_b[m])[0, 1]), 4),
            s32_fail_sim=round(fails["S32"] / max(visits["S32"], 1), 4), s32_fail_bosch=bosch["S32"]["fail_rate"],
            median_stations_sim=float(np.median([c[3] for c in done]))))
        r = res["runs"][-1]
        print(season, seed, {x: r[x] for x in ("throughput_per_h", "defect_rate", "breakdowns", "availability", "oee", "hvac_kwh",
                                              "share_corr", "share_max_abs", "failrate_corr", "s32_fail_sim", "median_stations_sim", "sec")}, flush=True)
# what-if demo on a July twin: maintenance before a predicted failure, and a stopped parallel machine
sim = LineSim(cfg, clim, start="2025-07-03 06:00", seed=7).run(26 * 60)
deg = [s for s, S in sim.S.items() if S["degraded"] and S["state"] == "up"]
target = deg[0] if deg else max(sim.S, key=lambda s: sim.S[s]["haz"] / sim.S[s]["haz_E"])
res["whatif"] = dict(
    time=str(sim.clock()), degraded_now=deg, target=target,
    no_action=sim.whatif([], 12), maintain_now=sim.whatif([dict(action="schedule_maintenance", station_id=target)], 12),
    s35_slow_50=sim.whatif([dict(action="set_speed", station_id="S35", pct=50)], 12),
    s35_slow_50_dispatch_s36=sim.whatif([dict(action="set_speed", station_id="S35", pct=50),
                                         dict(action="set_dispatch", weights={"S35": 0.4, "S36": 1.6})], 12),
    extra_cooling=sim.whatif([dict(action="set_hall_coupling", value=0.15)], 12),
    no_action_24h=sim.whatif([], 24), maintain_all_degraded_24h=sim.whatif([dict(action="schedule_maintenance", station_id=s, in_minutes=40 * i) for i, s in enumerate(deg)], 24))
for k, v in res["whatif"].items():
    if isinstance(v, dict):
        print("whatif", k, {x: f"{v['mean'][x]:.4g}+-{v['sd'][x]:.2g}" for x in ("throughput_per_h", "defect_rate", "breakdowns", "oee", "hvac_kwh", "cost")})
    else:
        print("whatif", k, v)
json.dump(res, open(out, "w"), indent=1, default=str)
