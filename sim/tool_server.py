"""smartbai tool server: exposes the twin to an LLM agent over HTTP (docs/adapter_interface.md). Standard library only.

  GET  /tools                      tool list with JSON schemas
  POST /tool/<name>   {args}       call a tool; every result carries observation_id and t_sim
  POST /admin/reset   {start, seed, warmup_h, params}   new twin (scenario set-up, not for the agent)
  POST /admin/advance {minutes}    run the twin forward (scenario clock, not for the agent)
  GET  /admin/observations         every observation issued since reset (for evidence checking)

predict_failure is a baseline that only uses observable sensors (vibration and motor temperature relative to what the
station's utilisation explains); it never reads the simulator's hidden failure state. It will be replaced by a model.
usage: python tool_server.py TWIN_V2_CONFIG CLIMATE_CSV [--port 8765] [--start "2025-07-03 06:00"] [--warmup-h 24]"""
import argparse
import itertools
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from line_sim import LineSim, load_climate  # noqa: E402

TOOLS = {
    "get_line_state": dict(description="State of every station, queues and the line KPIs since the start of the current shift view.", params={}),
    "get_station": dict(description="Live state of one station: status, speed, queue, motor temperature, vibration, failures so far.",
                        params={"station_id": "string, e.g. S32"}),
    "get_sensor_history": dict(description="Time series of one signal of one station.",
                               params={"station_id": "string", "signal": "motor_temp | vibration | queue | status", "hours": "number <= 48"}),
    "predict_failure": dict(description="Probability that the station fails within the horizon, from its recent sensor data (baseline model).",
                            params={"station_id": "string", "horizon_h": "number"}),
    "run_whatif": dict(description="Simulate a copy of the line forward with a list of actions (the real twin is not changed). Returns mean and sd of KPIs over replications.",
                       params={"actions": "list of {action: set_speed|schedule_maintenance|set_dispatch|set_hall_coupling, ...}", "hours": "number <= 48", "reps": "int <= 10 (default 5)"}),
    "schedule_maintenance": dict(description="Stop a station for preventive maintenance (30 min) after in_minutes.", params={"station_id": "string", "in_minutes": "number >= 0"}),
    "set_speed": dict(description="Set a station's speed in percent of nominal (30-150). Faster wears the machine more.", params={"station_id": "string", "pct": "number"}),
    "set_dispatch": dict(description="Split of parts between parallel stations, e.g. {S35: 0.5, S36: 1.5}.", params={"weights": "object station_id -> weight"}),
    "submit_decision": dict(description="Final decision of the case. Every number in the report must cite an observation_id.",
                            params={"report": "object: actions, rationale, claims: [{text, value, observation_id}]"}),
}


class Twin:
    def __init__(self, cfg, climate):
        self.cfg, self.climate, self.lock = cfg, climate, threading.Lock()
        self.reset({})

    def reset(self, a):
        self.sim = LineSim(self.cfg, self.climate, start=a.get("start", "2025-07-03 06:00"), seed=int(a.get("seed", 0)), params=a.get("params"))
        self.sim.run(float(a.get("warmup_h", 24)) * 60)
        self.base_t, self.base_c = self.sim.t, self.sim.counters()
        self.obs, self.counter, self.decision = {}, itertools.count(1), None
        return dict(ok=True, time=str(self.sim.clock()), t_sim=round(self.sim.t, 1))

    def observe(self, tool, args, result):
        oid = f"obs-{next(self.counter):05d}"
        rec = dict(observation_id=oid, t_sim=round(self.sim.t, 1), time=str(self.sim.clock()), tool=tool, args=args, result=result)
        self.obs[oid] = rec
        return rec

    # ---------------------------------------------------------------- tools
    def get_line_state(self, a):
        sim = self.sim
        return dict(stations=[sim.station_view(s) for s in sim.ids], outdoor_c=round(sim.outdoor_c(), 1), hall_c=round(sim.hall_c(), 1),
                    kpis_since_reset=sim.kpis(since=self.base_t, base=self.base_c))

    def get_station(self, a):
        return self.sim.station_view(self.sid(a))

    def get_sensor_history(self, a):
        h = min(float(a.get("hours", 6)), 48.0)
        return dict(station_id=self.sid(a), signal=a["signal"], hours=h, series=self.sim.sensor_history(self.sid(a), a["signal"], h))

    def predict_failure(self, a):
        sid, hz = self.sid(a), float(a.get("horizon_h", 4))
        S = self.sim.S[sid]
        vib = [v for _, v in self.sim.sensor_history(sid, "vibration", 1.0)]
        tmp = [v for _, v in self.sim.sensor_history(sid, "motor_temp", 1.0)]
        if not vib or S["state"] != "up":
            return dict(station_id=sid, horizon_h=hz, probability=None, note=f"station is {S['state']}")
        util = min(1.0, S["busy_time"] / max(self.sim.t, 1.0))
        excess_v = float(np.mean(vib)) - (0.9 + 0.6 * util)          # vibration not explained by load
        excess_t = float(np.mean(tmp)) - (self.sim.hall_c() + 14 * util)
        z = 4.0 * excess_v + 0.25 * excess_t - 1.5
        p_now = 1 / (1 + math.exp(-z))
        p = 1 - (1 - p_now) ** max(hz / 4.0, 0.25)
        return dict(station_id=sid, horizon_h=hz, probability=round(p, 3), evidence=dict(vibration_excess=round(excess_v, 3), motor_temp_excess_c=round(excess_t, 2)),
                    model="baseline: logistic on load-adjusted vibration and motor temperature (last hour)")

    def run_whatif(self, a):
        h, reps = min(float(a.get("hours", 12)), 48.0), int(min(int(a.get("reps", 5)), 10))
        w = self.sim.whatif(a.get("actions", []), h, reps=reps)
        return dict(hours=h, reps=reps, actions=a.get("actions", []), mean=w["mean"], sd=w["sd"])

    def schedule_maintenance(self, a):
        self.sim.schedule_maintenance(self.sid(a), float(a.get("in_minutes", 0)))
        return dict(ok=True, station_id=self.sid(a), in_minutes=float(a.get("in_minutes", 0)))

    def set_speed(self, a):
        self.sim.set_speed(self.sid(a), float(a["pct"]))
        return dict(ok=True, station_id=self.sid(a), speed_pct=round(self.sim.S[self.sid(a)]["speed"] * 100))

    def set_dispatch(self, a):
        w = {k: float(v) for k, v in a["weights"].items() if k in self.sim.S}
        self.sim.set_dispatch(w)
        return dict(ok=True, weights=w)

    def submit_decision(self, a):
        self.decision = a.get("report")
        return dict(ok=True, received=True)

    def sid(self, a):
        s = str(a.get("station_id", "")).upper()
        if s not in self.sim.S:
            raise ValueError(f"unknown station_id {s!r}; valid: {', '.join(self.sim.ids)}")
        return s


def make_handler(twin):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/tools":
                return self._send(200, TOOLS)
            if self.path == "/admin/observations":
                return self._send(200, dict(observations=list(twin.obs.values()), decision=twin.decision))
            self._send(404, dict(error="not found"))

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                args = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, dict(error=f"bad JSON: {e}"))
            with twin.lock:
                try:
                    if self.path == "/admin/reset":
                        return self._send(200, twin.reset(args))
                    if self.path == "/admin/advance":
                        twin.sim.run(float(args.get("minutes", 10)))
                        return self._send(200, dict(ok=True, time=str(twin.sim.clock()), t_sim=round(twin.sim.t, 1)))
                    if self.path.startswith("/tool/"):
                        name = self.path[len("/tool/"):]
                        if name not in TOOLS:
                            return self._send(404, dict(error=f"unknown tool {name}", tools=list(TOOLS)))
                        return self._send(200, twin.observe(name, args, getattr(twin, name)(args)))
                except (ValueError, KeyError, TypeError) as e:  # tool errors go back to the model as observations
                    return self._send(400, dict(error=f"{type(e).__name__}: {e}"))
            self._send(404, dict(error="not found"))
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("climate")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--start", default="2025-07-03 06:00")
    ap.add_argument("--warmup-h", type=float, default=24)
    a = ap.parse_args()
    twin = Twin(json.load(open(a.config)), load_climate(a.climate))
    twin.reset(dict(start=a.start, warmup_h=a.warmup_h))
    print(f"smartbai tool server on http://{a.host}:{a.port}  twin time {twin.sim.clock()}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(twin)).serve_forever()
