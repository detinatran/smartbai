"""Agents that drive the tool server over HTTP. An agent is a function agent(client, task) that calls tools and ends with
submit_decision. rule_agent is condition (c) of EVAL_PROTOCOL.md (fixed rules, read tools only, no run_whatif);
noop_agent only looks and submits "no action" (pipeline check). An LLM adapter plugs in with the same signature."""
import json
import urllib.error
import urllib.request


class ToolClient:
    def __init__(self, base_url, timeout=600):
        self.base, self.timeout = base_url.rstrip("/"), timeout

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read() or b"{}") | {"http_status": e.code}

    def call(self, tool, **args):
        return self._req("POST", f"/tool/{tool}", args)

    def task(self):
        return self._req("GET", "/task")

    def admin(self, path, body=None):
        return self._req("POST" if body is not None else "GET", f"/admin/{path}", body)


def noop_agent(client, task):
    s = client.call("get_line_state")
    client.call("submit_decision", report=dict(actions=[], rationale="no action (pipeline check)",
                                               claims=[dict(text="line state read", value=s["result"]["kpis_since_reset"]["completed"],
                                                            observation_id=s["observation_id"])]))


def rule_agent(client, task, p_fail=0.5, defect_mult=3.0, heat_c=42.0):
    claims, acts = [], []
    s = client.call("get_line_state")
    st = s["result"]["stations"]
    # 1) a stopped station with parallel siblings: send its parts to the siblings
    for v in st:
        if v["state"] != "up" and len(v["parallel_group"]) > 1:
            w = {g: (0.0 if g == v["station_id"] else 1.0) for g in v["parallel_group"]}
            hrs = max(1, -(-int(v.get("back_in_min", 60)) // 60))  # avoid it until the technician's estimate, rounded up
            client.call("set_dispatch", weights=w, hours=hrs)
            acts.append(f"avoid {v['station_id']} for {hrs} h")
            claims.append(dict(text=f"{v['station_id']} is {v['state']}", value=v["state"], observation_id=s["observation_id"]))
    # 2) highest predicted failure risk above the threshold: maintain it now
    risks = []
    for v in st:
        if v["state"] == "up":
            r = client.call("predict_failure", station_id=v["station_id"], horizon_h=4)
            pr = r.get("result", {}).get("probability")
            if pr is not None:
                risks.append((pr, v["station_id"], r["observation_id"]))
    risks.sort(reverse=True)
    maintained = set()
    if risks and risks[0][0] >= p_fail:
        pr, sid, oid = risks[0]
        client.call("schedule_maintenance", station_id=sid, in_minutes=0)
        maintained.add(sid)
        acts.append(f"maintain {sid}")
        claims.append(dict(text=f"{sid} failure probability within 4 h", value=pr, observation_id=oid))
    # 3) a station whose recent defect rate is >= 3x the line rate: maintain it (if not already)
    q = client.call("get_quality", hours=4)
    qr = q.get("result", {})
    line = qr.get("line_defect_rate", 0) or 0
    for row in qr.get("stations", []):
        if line > 0 and row["defect_rate"] >= defect_mult * line and row["parts"] >= 100 and row["station_id"] not in maintained:
            client.call("schedule_maintenance", station_id=row["station_id"], in_minutes=0)
            maintained.add(row["station_id"])
            acts.append(f"maintain {row['station_id']} (quality)")
            claims.append(dict(text=f"{row['station_id']} defect rate (4 h)", value=row["defect_rate"], observation_id=q["observation_id"]))
            break
    # 4) very hot outside: cooling high
    if s["result"]["outdoor_c"] >= heat_c:
        client.call("set_cooling", level="high")
        acts.append("cooling high")
        claims.append(dict(text="outdoor temperature (C)", value=s["result"]["outdoor_c"], observation_id=s["observation_id"]))
    # 5) rush order in the task: speed the trunk up to 115%
    if "customer needs at least" in task.lower():
        for sid in ("S29", "S30", "S33", "S34", "S35", "S36", "S37"):
            client.call("set_speed", station_id=sid, pct=115)
        acts.append("trunk speed 115%")
    client.call("submit_decision", report=dict(actions=acts, rationale="fixed rules: reroute stopped parallel stations, maintain the "
                                                                       "highest predicted risk >= 0.5 and any 3x defect station, cooling above 42 C, trunk +15% on rush orders",
                                               claims=claims))


AGENTS = dict(rule=rule_agent, noop=noop_agent)
