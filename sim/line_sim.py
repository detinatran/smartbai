"""smartbai discrete-event simulator for twin v2 (Bosch-derived line, Dubai climate).

From data:
  routing    Markov chain over stations from the Bosch transition counts (entry, branch and exit probabilities);
  load       each station's share of parts (sets its arrival rate);
  quality    each station's failure rate among visiting parts (excess over the line average is attributed to it);
  climate    hourly outdoor temperature in Dubai (Open-Meteo archive) drives the hall temperature.
Assumed (placeholders until partner data exist; all in ASSUMED and reported with every run):
  arrival rate, target utilisation, MTBF/MTTR, temperature sensitivity of the failure rate, sensor models, power and costs.

Time unit: minutes. Every station failure is preceded by a degradation window in which its motor temperature and
vibration drift upward and it produces more defects, so failure prediction has a real (simulated) signal."""
import copy
import heapq
import json
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ASSUMED = dict(
    arrival_per_min=6.0,        # parts entering the line per minute
    target_util=0.75,           # nominal utilisation of every station at the observed load
    service_cv=0.3,             # coefficient of variation of service times (lognormal)
    mtbf_h=60.0,                # mean operating hours between failures at 25 C hall temperature and speed 1.0
    mttr_min=45.0,              # mean repair time (lognormal, cv 0.5)
    temp_k=0.05,                # failure-rate sensitivity: x exp(temp_k * (T_hall - 25))
    hall_base_c=26.0,           # hall set point; hall = base + coupling * (outdoor - base)
    hall_coupling=0.30,
    precursor_h=4.0,            # degradation window before a failure (nominal hours)
    degraded_defect_mult=3.0,   # defect probability multiplier while degraded
    maint_min=30.0,             # duration of a preventive maintenance stop
    power_busy_kw=3.0, power_idle_kw=0.8,
    hvac_kw_per_deg=8.0,        # cooling power per degree the hall is kept below the outdoor temperature
    price_per_kwh=0.4,          # cost units
    defect_cost=25.0,           # cost units per defective part shipped to rework
    downtime_cost_per_h=300.0,  # cost units per station-hour of unplanned downtime
    value_per_part=10.0,        # cost units of value per completed part (net value = value - cost - shortfall)
    shortfall_cost_per_part=40.0,  # rush orders: penalty per part below the target
    cooling_high_coupling=0.15,    # hall coupling when cooling is set to "high"
    auto_reroute=False,         # False: parts follow the dispatch weights even to a stopped station (queue builds up)
    repair_est_sigma=0.0,       # noise of the technician's repair estimate (back_in_min): true x lognormal(0, sigma)
    sensor_period_min=10.0,
)


def load_climate(path):
    d = pd.read_csv(path, skiprows=3)
    d.columns = ["time", "temp_c", "rh", "dew_c", "sw_rad", "wind_kmh", "p_hpa"]
    d["time"] = pd.to_datetime(d["time"])
    return d.set_index("time")["temp_c"]


class LineSim:
    def __init__(self, cfg, climate, start="2025-07-15 06:00", seed=0, params=None):
        self.P = dict(ASSUMED, **(params or {}))
        self.cfg, self.climate = cfg, climate
        self.t0 = pd.Timestamp(start)
        self.rng = np.random.default_rng(seed)
        self.est_rng = np.random.default_rng([int(seed), 99])  # separate stream: estimates never change the trajectory
        self.t = 0.0
        self.ev, self.k = [], 0
        st = {s["id"]: s for s in cfg["stations"]}
        self.ids = list(st)
        avg = cfg["fail_rate"]
        # routing from transition counts
        out, inn = defaultdict(list), defaultdict(int)
        for e in cfg["edges"]:
            out[e["src"]].append((e["dst"], e["parts"]))
            inn[e["dst"]] += e["parts"]
        self.route = {}
        for s in self.ids:
            vis = st[s]["parts"]
            outs = out.get(s, [])
            tot = sum(c for _, c in outs)
            exit_p = max(0.0, 1 - tot / vis) if vis else 1.0
            dst = [d for d, _ in outs] + [None]
            p = np.array([c for _, c in outs] + [0.0], float)
            p = p / p.sum() * (1 - exit_p) if tot else p
            p[-1] = exit_p if tot else 1.0
            self.route[s] = (dst, p / p.sum())
        ent = {s: max(0.0, st[s]["parts"] - inn.get(s, 0)) for s in self.ids}
        z = sum(ent.values())
        self.entry = ([s for s in self.ids if ent[s] > 0], np.array([ent[s] / z for s in self.ids if ent[s] > 0]))
        self.entry_p = {s: ent[s] / z for s in self.ids if ent[s] > 0}
        # stations
        lam = self.P["arrival_per_min"]
        mean_visits = sum(st[s]["share_parts"] for s in self.ids)
        # calibrate: expected defects per part = line average. Station excess (fail rate above average) is attributed
        # to that station; the rest is spread over a typical path; degraded windows add their expected share.
        excess = sum(st[s]["share_parts"] * max(0.0, st[s]["fail_rate"] - avg) for s in self.ids)
        deg_frac = self.P["precursor_h"] / self.P["mtbf_h"]
        base_defect = max(avg - excess, 0.0) / max(mean_visits, 1) / (1 + deg_frac * (self.P["degraded_defect_mult"] - 1))
        self.S = {}
        self.parts_cfg = {s: st[s]["parts"] for s in self.ids}
        for s in self.ids:
            self.S[s] = dict(
                id=s, group=st[s]["parallel_group"], mean_service=self.P["target_util"] / max(lam * st[s]["share_parts"], 1e-6),
                p_defect=(base_defect + max(0.0, st[s]["fail_rate"] - avg)), speed=1.0, state="up", down_until=0.0,
                queue=deque(), busy=False, busy_since=0.0, busy_time=0.0, down_time=0.0, maint_time=0.0,
                haz=0.0, haz_E=self.rng.exponential(), degraded=False, failures=0, done=0, defects_made=0,
                weight=1.0, sensors=defaultdict(list), util_window=0.0, drift=1.0, stop_t=0.0, est_until=0.0)
        self.parts_in = 0
        self.completed = []  # (t_out, lead_time, defect, n_stations, path)
        self.log = []        # breakdowns, repairs, actions
        self.heat = None     # (t_start, t_end, delta_c): injected heat wave
        self.rush = None     # (t_start, t_end, target_parts): injected rush order
        self.base_coupling, self.cooling = self.P["hall_coupling"], "normal"
        self._last_sensor = 0.0
        self.push(self.rng.exponential(1 / lam), "arrive", None)
        self.push(0.0, "hour", None)
        self.push(0.0, "sensor", None)

    # ------------------------------------------------------------------ helpers
    def push(self, t, kind, data):
        self.k += 1
        heapq.heappush(self.ev, (t, self.k, kind, data))

    def clock(self, t=None):
        return self.t0 + pd.Timedelta(minutes=self.t if t is None else t)

    def outdoor_c(self, t=None):
        t = self.t if t is None else t
        ts = self.clock(t).floor("h")
        try:
            v = float(self.climate.loc[ts])
        except KeyError:
            v = float(self.climate.iloc[(self.climate.index.get_indexer([ts], method="nearest"))[0]])
        if self.heat and self.heat[0] <= t < self.heat[1]:
            v += self.heat[2]
        return v

    def hall_c(self, t=None):
        b = self.P["hall_base_c"]
        return b + self.P["hall_coupling"] * (self.outdoor_c(t) - b)

    def lognormal(self, mean, cv):
        s2 = math.log(1 + cv * cv)
        return float(self.rng.lognormal(math.log(mean) - s2 / 2, math.sqrt(s2)))

    # ------------------------------------------------------------------ part flow
    def enqueue(self, sid, part):
        S = self.S[sid]
        if S["state"] != "up" and self.P["auto_reroute"]:  # optional: route around a stopped machine
            alts = [g for g in S["group"] if g != sid and self.S[g]["state"] == "up"]
            if alts:
                sid = min(alts, key=lambda g: len(self.S[g]["queue"]))
                S = self.S[sid]
        S["queue"].append(part)
        if not S["busy"] and S["state"] == "up":
            self.start(sid)

    def start(self, sid):
        S = self.S[sid]
        if not S["queue"]:
            return
        part = S["queue"].popleft()
        S["busy"], S["busy_since"] = True, self.t
        self.push(self.t + self.lognormal(S["mean_service"] / S["speed"], self.P["service_cv"]), "done", (sid, part))

    def pick_next(self, sid):
        dst, p = self.route[sid]
        nxt = dst[self.rng.choice(len(dst), p=p)]
        if nxt is None:
            return None
        grp = self.S[nxt]["group"]
        if len(grp) > 1:  # within a parallel group use the dispatch weights (default = observed split)
            w = np.array([self.S[g]["weight"] * self.parts_cfg[g] for g in grp], float)
            if w.sum() > 0:
                nxt = grp[self.rng.choice(len(grp), p=w / w.sum())]
        return nxt

    # ------------------------------------------------------------------ main loop
    def run(self, minutes):
        end = self.t + minutes
        while self.ev and self.ev[0][0] <= end:
            t, _, kind, data = heapq.heappop(self.ev)
            self.t = t
            getattr(self, "_on_" + kind)(data)
        self.t = end
        return self

    def _on_arrive(self, _):
        self.parts_in += 1
        sid = self.entry[0][self.rng.choice(len(self.entry[0]), p=self.entry[1])]
        grp = [g for g in self.S[sid]["group"] if g in self.entry_p]
        if len(grp) > 1:  # parallel entry stations follow the dispatch weights too
            w = np.array([self.S[g]["weight"] * self.entry_p[g] for g in grp])
            if w.sum() > 0:
                sid = grp[self.rng.choice(len(grp), p=w / w.sum())]
        self.enqueue(sid, dict(id=self.parts_in, t_in=self.t, defect=False, n=0, path=[]))
        self.push(self.t + self.rng.exponential(1 / self.P["arrival_per_min"]), "arrive", None)

    def _on_done(self, data):
        sid, part = data
        S = self.S[sid]
        if S["state"] != "up":  # machine stopped mid-cycle: finish after it comes back
            self.push(S["down_until"] + 0.01, "done", data)
            return
        S["busy"] = False
        S["busy_time"] += self.t - S["busy_since"]
        S["done"] += 1
        p = S["p_defect"] * S["drift"] * (self.P["degraded_defect_mult"] if S["degraded"] else 1.0)
        if self.rng.random() < p:
            part["defect"] = True
            S["defects_made"] += 1
        part["n"] += 1
        part["path"].append(sid)
        nxt = self.pick_next(sid)
        if nxt is None:
            self.completed.append((self.t, self.t - part["t_in"], part["defect"], part["n"], tuple(part["path"])))
        else:
            self.enqueue(nxt, part)
        self.start(sid)

    def _on_hour(self, _):
        f_t = math.exp(self.P["temp_k"] * (self.hall_c() - 25.0))
        for S in self.S.values():
            if S["state"] != "up":
                continue
            S["haz"] += f_t * S["speed"] ** 2 / self.P["mtbf_h"]
            S["degraded"] = S["haz"] >= S["haz_E"] - self.P["precursor_h"] / self.P["mtbf_h"]
            if S["haz"] >= S["haz_E"]:
                rep = self.lognormal(self.P["mttr_min"], 0.5)
                self.stop(S, "down", rep)
                S["failures"] += 1
                self.log.append(dict(t=self.t, time=str(self.clock()), event="breakdown", station=S["id"], repair_min=round(rep, 1)))
        self.push(self.t + 60.0, "hour", None)

    def stop(self, S, state, minutes):
        S["state"], S["down_until"], S["stop_t"] = state, self.t + minutes, self.t
        sig = self.P["repair_est_sigma"]
        S["est_until"] = self.t + minutes * (float(self.est_rng.lognormal(0.0, sig)) if sig > 0 else 1.0)
        if S["busy"]:
            S["busy_time"] += self.t - S["busy_since"]
            S["busy_since"] = S["down_until"]
        self.push(S["down_until"], "restart", (S["id"], state, minutes))

    def _on_restart(self, data):
        sid, state, minutes = data
        S = self.S[sid]
        if S["down_until"] > self.t + 1e-9:
            return
        S["down_time" if state == "down" else "maint_time"] += minutes
        S["state"], S["haz"], S["haz_E"], S["degraded"], S["drift"] = "up", 0.0, self.rng.exponential(), False, 1.0
        self.log.append(dict(t=self.t, time=str(self.clock()), event="repaired" if state == "down" else "maintained", station=sid))
        if not S["busy"]:
            self.start(sid)

    def _on_maint(self, sid):
        S = self.S[sid]
        if S["state"] == "up":
            self.stop(S, "maint", self.P["maint_min"])
            self.log.append(dict(t=self.t, time=str(self.clock()), event="maintenance_start", station=sid))

    def _on_sensor(self, _):
        dt = self.P["sensor_period_min"]
        hall = self.hall_c()
        for S in self.S.values():
            busy = S["busy_time"] + ((self.t - S["busy_since"]) if S["busy"] else 0.0)
            util = min(1.0, max(0.0, (busy - S["util_window"]) / dt)) if self.t > 0 else 0.0
            S["util_window"] = busy
            prog = max(0.0, (S["haz"] - (S["haz_E"] - self.P["precursor_h"] / self.P["mtbf_h"])) / (self.P["precursor_h"] / self.P["mtbf_h"])) if S["degraded"] else 0.0
            on = S["state"] == "up"
            S["sensors"]["motor_temp"].append((self.t, round(hall + (14 * util if on else 0) + 9 * prog + self.rng.normal(0, 0.8), 2)))
            S["sensors"]["vibration"].append((self.t, round((0.9 + 0.6 * util if on else 0.05) + 1.4 * prog + self.rng.normal(0, 0.08), 3)))
            S["sensors"]["queue"].append((self.t, len(S["queue"])))
            S["sensors"]["status"].append((self.t, S["state"]))
        self.push(self.t + dt, "sensor", None)

    # ------------------------------------------------------------------ actions (used by the tool server and what-if)
    def set_speed(self, sid, pct):
        self.S[sid]["speed"] = max(0.3, min(1.5, pct / 100.0))
        self.log.append(dict(t=self.t, time=str(self.clock()), event="set_speed", station=sid, pct=pct))

    def schedule_maintenance(self, sid, in_minutes=0.0):
        self.push(self.t + max(0.0, in_minutes), "maint", sid)
        self.log.append(dict(t=self.t, time=str(self.clock()), event="schedule_maintenance", station=sid, in_minutes=in_minutes))

    def set_dispatch(self, weights, hours=None):
        for sid, w in weights.items():
            self.S[sid]["weight"] = max(0.0, float(w))
        if hours:  # temporary: back to the observed split after `hours`
            self.push(self.t + 60.0 * float(hours), "dispatch_revert", list(weights))
        self.log.append(dict(t=self.t, time=str(self.clock()), event="set_dispatch", weights=weights, hours=hours))

    def _on_dispatch_revert(self, sids):
        for sid in sids:
            self.S[sid]["weight"] = 1.0
        self.log.append(dict(t=self.t, time=str(self.clock()), event="dispatch_revert", stations=sids))

    def apply(self, actions):
        for a in actions or []:
            kind = a["action"]
            if kind == "set_speed":
                self.set_speed(a["station_id"], a["pct"])
            elif kind == "schedule_maintenance":
                self.schedule_maintenance(a["station_id"], a.get("in_minutes", 0.0))
            elif kind == "set_dispatch":
                self.set_dispatch(a["weights"], a.get("hours"))
            elif kind == "set_hall_coupling":  # e.g. extra cooling
                self.P["hall_coupling"] = float(a["value"])
            elif kind == "set_cooling":
                lvl = a.get("level", "normal")
                self.P["hall_coupling"] = self.P["cooling_high_coupling"] if lvl == "high" else self.base_coupling
                self.cooling = lvl
                self.log.append(dict(t=self.t, time=str(self.clock()), event="set_cooling", level=lvl))
            else:
                raise ValueError(f"unknown action {kind}")

    def inject(self, ev):
        """scenario events (hidden from the agent; it has to find them through the tools)."""
        kind = ev["type"]
        if kind == "degrade":
            S = self.S[ev["station"]]
            S["haz"] = S["haz_E"] - 0.9 * self.P["precursor_h"] / self.P["mtbf_h"]
            S["degraded"] = True
        elif kind == "breakdown":
            S = self.S[ev["station"]]
            if S["state"] == "up":
                self.stop(S, "down", float(ev["repair_min"]))
                S["failures"] += 1
                self.log.append(dict(t=self.t, time=str(self.clock()), event="breakdown", station=S["id"], repair_min=float(ev["repair_min"])))
        elif kind == "defect_drift":
            self.S[ev["station"]]["drift"] = float(ev["mult"])
        elif kind == "heat":
            self.heat = (self.t, self.t + 60.0 * float(ev["hours"]), float(ev["delta_c"]))
        elif kind == "rush":
            self.rush = (self.t, self.t + 60.0 * float(ev["hours"]), float(ev["target_parts"]))
        else:
            raise ValueError(f"unknown event {kind}")

    def quality_by_station(self, hours, min_parts=30):
        """defect rate among the parts completed in the last `hours`, by station visited (as in the Bosch analysis)."""
        lo = self.t - hours * 60.0
        vis, bad = defaultdict(int), defaultdict(int)
        done = [c for c in self.completed if c[0] >= lo]
        for c in done:
            for s in set(c[4]):
                vis[s] += 1
                bad[s] += c[2]
        line = sum(c[2] for c in done) / max(len(done), 1)
        rows = [dict(station_id=s, parts=vis[s], defect_rate=round(bad[s] / vis[s], 4)) for s in self.ids if vis[s] >= min_parts]
        return dict(hours=hours, parts=len(done), line_defect_rate=round(line, 4), stations=sorted(rows, key=lambda r: -r["defect_rate"]))

    def whatif(self, actions, hours, reps=5, seed_offset=0):
        """run `reps` copies forward with the given actions; the twin itself is not changed. Replication r uses the same
        random stream for every scenario asked at the same twin time (common random numbers), so differences between
        scenarios are not just noise. Returns mean, sd and per-replication KPIs."""
        runs = []
        for r in range(reps):
            sim = copy.deepcopy(self)
            sim.rng = np.random.default_rng([int(self.t * 1000), seed_offset + r])
            t_start, base = sim.t, sim.counters()
            sim.apply(actions)
            sim.run(hours * 60.0)
            runs.append(sim.kpis(since=t_start, base=base))
        keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
        return dict(hours=hours, reps=reps, actions=actions,
                    mean={k: round(float(np.mean([x[k] for x in runs])), 5) for k in keys},
                    sd={k: round(float(np.std([x[k] for x in runs], ddof=1)), 5) for k in keys} if reps > 1 else None,
                    runs=runs)

    def counters(self):
        return dict(busy=sum(S["busy_time"] for S in self.S.values()), down=sum(S["down_time"] for S in self.S.values()),
                    maint=sum(S["maint_time"] for S in self.S.values()))

    # ------------------------------------------------------------------ observations
    def kpis(self, since=0.0, base=None):
        """KPIs over [since, now]; base = counters() taken at `since` (zeros when since = 0)."""
        P = self.P
        b = base or dict(busy=0.0, down=0.0, maint=0.0)
        c = self.counters()
        done = [x for x in self.completed if x[0] >= since]
        hrs = max((self.t - since) / 60.0, 1e-9)
        n = len(done)
        defects = sum(x[2] for x in done)
        busy = (c["busy"] - b["busy"]) / 60.0
        st_h = len(self.S) * hrs
        down_h = (c["down"] - b["down"]) / 60.0
        maint_h = (c["maint"] - b["maint"]) / 60.0
        hours = pd.date_range(self.clock(since).floor("h"), self.clock().floor("h"), freq="h")
        out_c = np.array([float(self.climate.get(h, np.nan)) for h in hours])
        if self.heat:  # injected heat wave
            hm = np.array([(h - self.t0).total_seconds() / 60.0 for h in hours])
            out_c = out_c + np.where((hm >= self.heat[0] - 60) & (hm < self.heat[1]), self.heat[2], 0.0)
        hall_c = P["hall_base_c"] + P["hall_coupling"] * (out_c - P["hall_base_c"])
        hvac = float(np.nansum(np.maximum(0.0, out_c - hall_c)) * P["hvac_kw_per_deg"])  # heat removed ~ outdoor - hall
        machine = busy * P["power_busy_kw"] + max(st_h - busy - down_h - maint_h, 0) * P["power_idle_kw"]
        energy = machine + hvac
        nominal = P["arrival_per_min"] * 60.0
        availability = 1.0 - (down_h + maint_h) / max(st_h, 1e-9)
        cost = energy * P["price_per_kwh"] + defects * P["defect_cost"] + down_h * P["downtime_cost_per_h"]
        shortfall = 0.0
        if self.rush and since <= self.rush[0] + 1e-6 and self.t >= self.rush[1] - 1e-6:
            got = sum(1 for x in done if self.rush[0] <= x[0] <= self.rush[1])
            shortfall = max(0.0, self.rush[2] - got)
        net = n * P["value_per_part"] - cost - shortfall * P["shortfall_cost_per_part"]
        return dict(
            hours=round(hrs, 2), throughput_per_h=round(n / hrs, 1), completed=n, defect_rate=round(defects / max(n, 1), 5),
            mean_lead_time_min=round(float(np.mean([x[1] for x in done])) if n else float("nan"), 2),
            wip=int(sum(len(S["queue"]) + S["busy"] for S in self.S.values())),
            availability=round(availability, 4), performance=round(min(1.0, n / hrs / nominal), 4),
            oee=round(availability * min(1.0, n / hrs / nominal) * (1 - defects / max(n, 1)), 4),
            breakdowns=sum(1 for e in self.log if e["event"] == "breakdown" and e["t"] >= since),
            energy_kwh=round(energy, 1), hvac_kwh=round(hvac, 1),
            cost=round(cost, 1), shortfall_parts=round(shortfall, 1), net_value=round(net, 1),
            outdoor_c_mean=round(float(np.nanmean(out_c)), 1) if len(out_c) else None)

    def station_view(self, sid):
        S = self.S[sid]
        last = {k: v[-1][1] for k, v in S["sensors"].items() if v and k in ("motor_temp", "vibration")}
        back = dict(back_in_min=max(0, round(S["est_until"] - self.t))) if S["state"] != "up" else {}  # technician's repair estimate
        return dict(station_id=sid, state=S["state"], speed_pct=round(S["speed"] * 100), queue=len(S["queue"]), busy=S["busy"],
                    parts_done=S["done"], failures=S["failures"], parallel_group=S["group"], **back, **last)

    def sensor_history(self, sid, signal, hours):
        lo = self.t - hours * 60.0
        return [(round(t, 1), v) for t, v in self.S[sid]["sensors"][signal] if t >= lo]


def build_from_spec(cfg, climate, spec):
    """line state at the decision time of a case spec: warm-up, then the injected events. An event with lead_min > 0 is
    injected that many minutes before the decision time, so its effects can show in the observations (CERTIFY_PROTOCOL
    amendment 1); events without lead_min are injected at the decision time, as in EVAL_PROTOCOL."""
    sim = LineSim(cfg, climate, start=spec["start"], seed=int(spec["seed"]), params=spec.get("params"))
    end = float(spec.get("warmup_h", 24)) * 60.0
    for ev in sorted(spec.get("inject", []), key=lambda e: -float(e.get("lead_min", 0))):
        sim.run(max(end - float(ev.get("lead_min", 0)) - sim.t, 0.0))
        sim.inject(ev)
    sim.run(max(end - sim.t, 0.0))
    return sim

