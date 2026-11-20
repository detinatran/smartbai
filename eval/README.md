# Decision benchmark

The benchmark follows [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md), which was fixed before the oracle or any agent was run,
plus amendment 1.

There are 30 cases in 5 families: degradation, breakdown, quality drift, heat wave and rush order. Each case is a twin
state with one hidden injected event. An agent investigates through the tool server, takes actions, and submits a
decision with evidence. It is scored against an oracle that simulates every candidate action list.

| File | Role |
|---|---|
| [`scenarios.py`](scenarios.py) | builds the 30 cases (deterministic) |
| [`common.py`](common.py) | case state, the 164 candidate action lists, protocol evaluation (12 h, 8 replications, common random numbers), acceptable set |
| [`oracle.py`](oracle.py) | evaluates every candidate on every case in parallel and saves the oracle table |
| [`score.py`](score.py) | decision score (acceptable, regret, normalised regret) and evidence flags (`unsupported`, `mismatch`, `unverified_action`) |
| [`agents.py`](agents.py) | `ToolClient`, the fixed-rule agent (condition c) and a no-action agent |
| [`run_cases.py`](run_cases.py) | runs an agent over the cases through the tool server and writes one scored JSON line per case |

## Running

```bash
python eval/scenarios.py twin_v2_config.json dubai_hourly_2015_2025.csv scenarios.json
python eval/oracle.py twin_v2_config.json dubai_hourly_2015_2025.csv scenarios.json oracle.json --workers 12
python sim/tool_server.py twin_v2_config.json dubai_hourly_2015_2025.csv --port 8765 &
python eval/run_cases.py --server http://127.0.0.1:8765 --agent rule --condition c \
    --config twin_v2_config.json --climate dubai_hourly_2015_2025.csv --scenarios scenarios.json --oracle oracle.json --out rule.jsonl
```

`run_cases.py` resumes: cases already present in the output file are skipped.

## Plugging in an LLM agent

An agent is a function `agent(client, task)`:
- `client.call(tool, **args)` calls a tool and returns `{observation_id, t_sim, time, tool, args, result}`, or `{error, http_status}`;
- `task` is the case text, also available from `GET /task`, which lists the tools allowed in the case.

The agent must end with `submit_decision`:

```python
client.call("submit_decision", report={
    "actions": ["avoid S5 for 4 h"],
    "rationale": "...",
    "claims": [{"text": "S5 is down", "value": "down", "observation_id": "obs-00003"},
               {"text": "what-if net value", "value": 36755.0, "observation_id": "obs-00007"}]})
```

Run it with `--agent mymodule:my_agent` and `--condition a` (all tools) or `--condition b` (no `run_whatif`).

- **Scored actions:** only the actions issued through action tools (`schedule_maintenance`, `set_speed`, `set_dispatch`, `set_cooling`) are re-simulated and scored. The `actions` text in the report is free-form.
- **Scored claims:** every claim in the report is checked against the observation log.

## Baseline results (condition c)

First run of the fixed-rule agent and the no-action agent on the 30 cases. Regret is in net-value units over the 12 h
horizon; acceptable means the issued actions fall in the oracle's acceptable set.

| Family | Acceptable, no action | Acceptable, rules | Mean regret, no action | Mean regret, rules |
|---|---|---|---|---|
| degradation | 2/6 | 3/6 | 2,028 | 1,427 |
| breakdown | 2/6 | 5/6 | 2,231 | 1,234 |
| quality drift | 2/6 | 3/6 | 1,580 | 1,349 |
| heat wave | 3/6 | 1/6 | 1,388 | 1,965 |
| rush order | 2/6 | 3/6 | 4,781 | 5,841 |
| **all** | **11/30** | **15/30** | **2,401** | **2,363** |

- Rules help on breakdowns, degradation and quality drift.
- Rules hurt on heat waves and rush orders:
  - "Cooling high above 42 °C": cooling alone beats no action in only 3 of the 6 heat cases (from −1,181 to +974 net value).
  - "Trunk at 115% for a rush order" is worse than no action in 4 of the 6 rush cases (down to −4,959). The oracle's best is usually a maintenance, often with cooling.
- Over the 30 cases the two agents do not differ (paired Wilcoxon on regret, p = 0.24, exploratory).
- In condition c every issued action counts as unverified, by construction: `run_whatif` is disabled. The flag is meant for condition a.
