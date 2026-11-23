# Twin-certified selective decisions

This directory implements [`../CERTIFY_PROTOCOL.md`](../CERTIFY_PROTOCOL.md).

1. A proposer suggests up to 5 action lists.
2. The twin tests them against "no action", with paired replications and a budget of 64 runs.
3. The best one is executed on the plant only if its lower bound clears a threshold τ.
4. τ is calibrated so that the rate of actions that do worse than doing nothing on the plant stays below α = 0.10.

The twin differs from the plant in two ways:
- its physical parameters are nominal, while the plant's are perturbed by up to ×1.5;
- it sees only observables: the technician's repair estimate, a temperature forecast with bias, and sampled hidden
  degradation (see [`../../sim/twin_sync.py`](../../sim/twin_sync.py)).

| File | Role |
|---|---|
| [`../scenarios_certify.py`](../scenarios_certify.py) | cases per split (`--version`; v5 and v6: dev 36, calibration 600, test 300); six families, including "quiet" |
| [`../../sim/fit_failure_model.py`](../../sim/fit_failure_model.py) | the twin's degradation model, fitted on nominal simulations |
| [`proposers.py`](proposers.py) | rule and random-5 proposers; CLI that collects LLM proposals through the tool server (`--shard`, `--nshards`) |
| [`llm_agent.py`](llm_agent.py) | the LLM proposer (local open-weight model through llama-server); frozen in [`FROZEN.md`](FROZEN.md) |
| [`rs.py`](rs.py) | sequential paired comparison against no action, in the twin |
| [`ltt.py`](ltt.py) | Learn-then-Test threshold, pooled or one per family (amendment 5) |
| [`run.py`](run.py) | runs a split: twin testing per proposer, plant harm labels, the rule agent executed directly |
| [`summarize.py`](summarize.py) | calibrates τ (`--gate family` by default, `--gate pooled` for v1–v4), reports the conditions, tests the hypotheses |

## Running

```bash
python sim/fit_failure_model.py twin_v2_config.json dubai.csv failure_model.json
python eval/scenarios_certify.py twin_v2_config.json dubai.csv cases.json
python eval/certify/run.py twin_v2_config.json dubai.csv cases.json failure_model.json --split calibration --out cal.jsonl
python eval/certify/run.py twin_v2_config.json dubai.csv cases.json failure_model.json --split test --out test.jsonl
python eval/certify/summarize.py --calibration cal.jsonl --test test.jsonl
```

## Plugging in an LLM proposer

The proposer is an agent `agent(client, task)`. It runs against the tool server with **read tools only**: the action
tools and `run_whatif` return 403. It must end with:

```python
client.call("submit_decision", report={
    "candidates": [
        [{"action": "set_dispatch", "weights": {"S5": 0, "S4": 1}, "hours": 4}],
        [{"action": "schedule_maintenance", "station_id": "S30", "in_minutes": 0}],
        [{"action": "schedule_maintenance", "station_id": "S30", "in_minutes": 0}, {"action": "set_cooling", "level": "high"}]],
    "rationale": "...",
    "claims": [{"text": "S5 is down", "value": "down", "observation_id": "obs-00003"}]})
```

Up to 5 candidates. Each candidate is a list of actions.

| Action | Fields |
|---|---|
| `schedule_maintenance` | `station_id`, `in_minutes` |
| `set_speed` | `station_id`, `pct` (30–150) |
| `set_dispatch` | `weights` (station → weight), `hours` (optional) |
| `set_cooling` | `level`: `normal` or `high` |

Invalid actions are dropped and counted.

Collect proposals for a split, then run it with them:

```bash
python sim/tool_server.py twin_v2_config.json dubai.csv --port 8765 &
python eval/certify/proposers.py --server http://127.0.0.1:8765 --agent mymodule:my_agent --cases cases.json --split dev --out llm_dev.jsonl
python eval/certify/run.py ... --split dev --llm-proposals llm_dev.jsonl --out dev_llm.jsonl
```

Under the protocol:
- the model, prompt and decoding settings are frozen in `FROZEN.md` before the calibration split is collected;
- only the dev split may be used to develop them.
