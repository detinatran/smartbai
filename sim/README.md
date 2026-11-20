# Discrete-event simulator (twin v2)

[`line_sim.py`](line_sim.py) simulates the twin v2 line in minutes. It needs only Python, numpy and pandas; one simulated
week takes about 9 s of CPU.

## What comes from data and what is assumed

| Component | Source |
|---|---|
| Part routing (entry, branch and exit probabilities) | transition counts between stations in the Bosch data (1.18 million parts) |
| Load per station | share of parts visiting the station (Bosch) |
| Defects | failure rate per station (Bosch). The excess over the line average is attributed to that station, and the rest is calibrated so the overall rate matches the data. |
| Outdoor temperature | hourly Dubai climate 2015–2025 (Open-Meteo) |
| Arrival rate, nominal utilisation, MTBF/MTTR, temperature sensitivity of the failure rate, sensor models, power, costs | **assumed**; listed in `ASSUMED` and reported with every run, to be replaced by partner data |

A stopped station keeps receiving its share of parts (its queue grows) until it is repaired or the dispatch is changed (`auto_reroute = False`).
Every breakdown is preceded by a **degradation window** (4 nominal hours by default). In that window motor temperature
and vibration drift upward and the station produces more defects, so failure prediction has a signal to learn from.
The failure rate rises with hall temperature, which depends on the Dubai outdoor temperature and on how hard the hall is
cooled.

## API (used by the tool server)

```python
from line_sim import LineSim, load_climate
sim = LineSim(cfg, load_climate(csv), start="2025-07-15 06:00", seed=0)
sim.run(minutes)                       # advance the twin
sim.kpis()                             # output/h, defect rate, lead time, WIP, availability, OEE, breakdowns, kWh, cost
sim.station_view("S32")                # state, speed, queue, motor temperature, vibration
sim.sensor_history("S32", "vibration", hours=6)
sim.apply([{"action": "set_speed", "station_id": "S35", "pct": 80}])          # real action on the twin
sim.whatif([{"action": "schedule_maintenance", "station_id": "S2"}], hours=24, reps=8)  # trial run; twin unchanged
```

Actions:
- `set_speed`;
- `schedule_maintenance`;
- `set_dispatch` (load split between parallel stations; optional `hours` after which the normal split returns);
- `set_hall_coupling` (more or less cooling).

`whatif` runs several replications. Every scenario asked at the same twin time uses the same random streams (common
random numbers), and the result is the mean ± sd.

## Validation ([`validate_sim.py`](validate_sim.py))

7 days × 3 seeds, July and January 2025 (current simulator, `auto_reroute = False`):

| | Simulator | Bosch |
|---|---|---|
| Correlation of per-station visit shares | 1.0 (max abs. error 1.0–1.3 pp) | |
| Correlation of per-station failure rates | 0.97–0.99 | |
| Overall failure rate | 0.57–0.63% | 0.58% |
| Failure rate at S32 | 3.7–6.0% | 4.51% |
| Median stations per part | 13 | 13 |

Climate effect, with the assumed temperature sensitivity `temp_k = 0.05`:
- July has about 23% more breakdowns than January (116–136 against 95–107 per week).
- HVAC uses about 6,800 kWh/week in July and almost nothing in January.

What-if examples (twin at 2025-07-04 08:00, 6 stations degraded, 5 replications):

| Scenario | Output/h | Breakdowns | HVAC (kWh) | Cost |
|---|---|---|---|---|
| No action (24 h) | 357 ± 4 | 25.0 ± 2.4 | 1,118 | 8,177 ± 500 |
| Maintain the 6 degraded stations one after another (24 h) | 355 ± 4 | 19.4 ± 0.6 | 1,118 | 6,654 ± 740 |
| No action (12 h) | 351 ± 7 | 15.4 ± 1.1 | 766 | 4,599 ± 370 |
| Extra cooling (12 h) | 346 ± 9 | 14.2 ± 1.6 | 930 | 4,439 ± 220 |
| S35 slowed to 50% (12 h) | 306 ± 5 | 14.0 ± 0.7 | 766 | 4,158 ± 500 |
| S35 at 50% + shift load to S36 (12 h) | 314 ± 5 | 15.6 ± 0.6 | 766 | 4,462 ± 290 |

These are the trade-offs the agent has to weigh:
- Maintenance reduces breakdowns and cost, but costs some output while the machines are stopped.
- Extra cooling reduces breakdowns, but raises HVAC energy by about 21%.
- Shifting load away from a slow station recovers output, but loads its sibling.

The full decision benchmark is in [`../eval`](../eval).

## Tool server ([`tool_server.py`](tool_server.py))

Exposes the twin to an LLM agent over HTTP, following [`../docs/adapter_interface.md`](../docs/adapter_interface.md).
It uses the standard library only.

```bash
python sim/tool_server.py TWIN_V2_CONFIG.json dubai_hourly_2015_2025.csv --port 8765 --start "2025-07-03 06:00"
curl localhost:8765/tools
curl -X POST localhost:8765/tool/get_station -d '{"station_id": "S32"}'
curl -X POST localhost:8765/tool/run_whatif -d '{"actions": [{"action": "schedule_maintenance", "station_id": "S2"}], "hours": 12}'
curl localhost:8765/admin/observations      # observation log for evidence checking
```

- Every tool result carries an `observation_id`, `t_sim` and the twin time.
- Tool errors (unknown station, missing argument) are returned so the model reads them as an observation.
- `/admin/reset` and `/admin/advance` are for whoever sets up the evaluation scenarios; the agent does not use them.

`predict_failure` is a **baseline** for now: a logistic score on load-adjusted vibration and motor temperature. It uses
only observable sensors and never reads the simulator's hidden failure state. On one simulated July week it reached
AUC = 0.785 (6,415 hourly samples, 458 of them followed by a breakdown within 4 hours). A learned model (layer 3) will
replace it.
