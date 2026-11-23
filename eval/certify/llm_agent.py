"""LLM proposer of CERTIFY_PROTOCOL.md: an open-weight model served by llama.cpp's OpenAI-compatible server
(llama-server). It reads the line through the read tools of the tool server and ends with submit_decision, returning
up to K = 5 candidate action lists. It never acts on the twin: action tools and run_whatif are disabled (403).

The model file, server build, prompt and decoding settings are frozen in FROZEN.md before a calibration split is run.

usage: proposers.py --agent llm_agent:agent ...   (env LLM_URL, default http://127.0.0.1:8080)"""
import json
import os
import urllib.error
import urllib.request

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080").rstrip("/")
MAX_TOOL_TURNS = 10  # model turns that may call read tools; one more turn is then given for submit_decision only
MAX_RESULT_CHARS = 8000  # tool results are compact JSON, cut to this length
DECODING = dict(temperature=0.0, top_p=1.0, seed=0, max_tokens=4096)
THINKING = os.environ.get("LLM_THINKING", "0") == "1"

SYSTEM = """You advise the supervisor of a production line with 39 stations (S0-S38). Some stations work in parallel
groups. You can read the line through tools, but you cannot change it. Your job is to propose up to 5 candidate plans
for the next 12 hours. Every candidate will be tested in a simulator of the line against taking no action, and only a
candidate that the simulator finds better than no action may be executed. The objective is net value over 12 hours:
value of completed parts minus energy, defect and unplanned-downtime costs (and, for a rush order, the cost of every
part short). Taking no action is always possible and often hard to beat, so every action must pay for its cost.

Actions and what they cost:
- {"action": "schedule_maintenance", "station_id": "S30", "in_minutes": 0}: stops the station for about 30 min of
  preventive maintenance and lowers the risk of a breakdown of a worn or degraded machine. An unplanned breakdown
  usually costs much more than a planned stop, so maintaining every station with a high failure probability or clear
  signs of wear (degraded state, rising vibration or motor temperature) can pay off; maintaining a station without
  such signs only loses output.
- {"action": "set_speed", "station_id": "S12", "pct": 115}: speed in percent of nominal (30-150). Faster raises the
  output of a station that limits the line (long queue in front, little behind) but wears and heats it; slower protects
  a hot or degraded machine and lowers its defects, at the cost of output.
- {"action": "set_dispatch", "weights": {"S35": 0, "S36": 1}, "hours": 4}: split of parts between the stations of one
  parallel group; hours is optional (revert to the normal split after that time). Use it to send parts away from a
  station that is down, degraded or making defects, toward its partners.
- {"action": "set_cooling", "level": "high"}: hall cooling. High keeps machines cooler but costs energy for all 12 hours;
  it pays only when the hall or the motors are hot.

How to work: read the line state, then look closer at what is unusual: stations that are down or degraded, hot or
vibrating motors, long queues, failure probabilities, defect rates (get_quality). Use at most about 8 tool calls, then
call submit_decision.

Candidates: the simulator picks among your candidates, so offer a spread rather than five versions of one plan. For
example: one broad plan that treats every problem you found; one or two narrower plans that treat only the most severe
problems; and one or two plans of a different kind (speed, dispatch or cooling). Include at least one candidate with a
single action. Order them from the one you believe best to the least. Keep the rationale short. In claims, cite the
observation_id of each number you rely on."""

ACTION = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["schedule_maintenance", "set_speed", "set_dispatch", "set_cooling"]},
        "station_id": {"type": "string"},
        "in_minutes": {"type": "number"},
        "pct": {"type": "number"},
        "weights": {"type": "object", "additionalProperties": {"type": "number"}},
        "hours": {"type": "number"},
        "level": {"type": "string", "enum": ["normal", "high"]},
    },
    "required": ["action"],
}


def fn(name, description, properties=None, required=None):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": required or []}}}


SID = {"type": "string", "description": "station id, e.g. S32"}
READ_TOOLS = [
    fn("get_line_state", "State of every station (state, speed, queue, parts done, failures, motor temperature, "
       "vibration, parallel group), hall and outdoor temperature, and line KPIs."),
    fn("get_station", "Live state of one station.", {"station_id": SID}, ["station_id"]),
    fn("get_sensor_history", "Time series of one signal of one station (minutes, value).",
       {"station_id": SID, "signal": {"type": "string", "enum": ["motor_temp", "vibration", "queue", "status"]},
        "hours": {"type": "number", "description": "<= 48"}}, ["station_id", "signal"]),
    fn("predict_failure", "Probability that the station fails within the horizon, from its recent sensor data.",
       {"station_id": SID, "horizon_h": {"type": "number"}}, ["station_id"]),
    fn("get_quality", "Defect rate among recently completed parts, by station visited.",
       {"hours": {"type": "number", "description": "<= 24"}}),
]
SUBMIT = fn("submit_decision", "Final answer: up to 5 candidate plans, each a list of actions.", {
    "candidates": {"type": "array", "maxItems": 5, "items": {"type": "array", "minItems": 1, "items": ACTION}},
    "rationale": {"type": "string"},
    "claims": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string"}, "value": {}, "observation_id": {"type": "string"}}, "required": ["text", "observation_id"]}},
}, ["candidates", "rationale"])


def chat(messages, tools):
    body = dict(model="local", messages=messages, tools=tools, tool_choice="auto", **DECODING,
                chat_template_kwargs=dict(enable_thinking=THINKING))
    req = urllib.request.Request(LLM_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"], out.get("usage", {})


def compact(x):
    s = json.dumps(x, separators=(",", ":"), default=str)
    return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + "...(cut)"


def agent(client, task, log=None):
    """runs one case; returns a trace dict (turns, tokens, tool calls). The decision is stored by submit_decision."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    trace = dict(turns=0, prompt_tokens=0, completion_tokens=0, calls=[], submitted=False, nudges=0)
    for turn in range(MAX_TOOL_TURNS + 1):
        final = turn == MAX_TOOL_TURNS
        if final:
            messages.append({"role": "user", "content": "Call submit_decision now with your candidates."})
        try:
            msg, usage = chat(messages, [SUBMIT] if final else READ_TOOLS + [SUBMIT])
        except urllib.error.HTTPError as e:  # llama-server returns 500 when a tool call's JSON is cut off or malformed
            if e.code != 500:
                raise
            trace["turns"] += 1
            trace["server_errors"] = trace.get("server_errors", 0) + 1
            messages.append({"role": "user", "content": "Your last tool call was malformed or cut off. Call the tool "
                             "again with valid JSON and a shorter rationale."})
            continue
        trace["turns"] += 1
        trace["prompt_tokens"] += usage.get("prompt_tokens", 0)
        trace["completion_tokens"] += usage.get("completion_tokens", 0)
        calls = msg.get("tool_calls") or []
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v is not None})
        if not calls:
            if final:
                break
            trace["nudges"] += 1
            messages.append({"role": "user", "content": "Use the tools, then call submit_decision."})
            continue
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args, name = {}, name + ":bad_json"
            trace["calls"].append(name)
            if name == "submit_decision":
                rep = dict(candidates=args.get("candidates", []), rationale=args.get("rationale", ""), claims=args.get("claims", []))
                client.call("submit_decision", report=rep)
                trace["submitted"] = True
                return trace
            r = client.call(name, **args) if name in {t["function"]["name"] for t in READ_TOOLS} else dict(error=f"unknown tool {name}")
            content = compact(dict(observation_id=r["observation_id"], result=r["result"]) if "observation_id" in r else r)
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": content})
    client.call("submit_decision", report=dict(candidates=[], rationale="no decision submitted", claims=[]))
    return trace
