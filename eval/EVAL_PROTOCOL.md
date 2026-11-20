# smartbai evaluation protocol (fixed 2026-11-20, before the oracle or any agent is run)

## Question
Does an LLM agent that can test actions in the twin (`run_whatif`) and must cite twin observations make better decisions
and fabricate less than (b) the same agent without `run_whatif` and (c) fixed rules?

## Scenarios (eval/scenarios.py, 30 cases, deterministic)
Each case is a twin state plus one injected event. The agent never sees the injected event; it has to find it through
the tools. All cases start on July days of 2025 in Dubai (hot season), 24 h warm-up, then the event, then the decision.

| Family | n | Injected event | Task given to the agent |
|---|---|---|---|
| degradation | 6 | one station enters its pre-failure degradation window | generic: decide actions for the next 12 h |
| breakdown | 6 | one station that has a parallel sibling stops for a long repair (180-300 min) | generic |
| quality | 6 | one station's defect probability is multiplied by 6 until it is maintained | generic |
| heat | 6 | outdoor temperature +6 C for the next 12 h | generic |
| rush | 6 | a customer needs at least N parts in the next 12 h (N = 1.04 x the no-action expectation) | generic + the target N |

Stations for degradation, quality and breakdown are drawn with a fixed seed: degradation and quality from all 39
stations weighted by flow, breakdown from stations that have a parallel sibling.

## Objective (identical for oracle, rules and agents)
Net value over the 12 h after the decision:
- value of completed parts;
- minus the cost of energy, defects and unplanned downtime;
- minus a shortfall penalty in rush cases.

All parameters are the simulator's `ASSUMED` values, frozen with this protocol.

## Action space and oracle (eval/oracle.py)
Actions: `schedule_maintenance` (now), `set_dispatch` (avoid one parallel station), `set_cooling` (high),
`set_speed` (bundles: trunk 115%, all 115%, all 85%), and pairs maintenance(s) + avoid(s) and maintenance(s) + cooling.
The candidate list is the same for every case: 137 candidates including "no action" (164 after amendment 1).

Each candidate is simulated for 12 h with 8 replications and common random numbers (the same streams for every candidate
of a case).

A candidate is **acceptable** if its paired mean shortfall to the best candidate is at most
max(2 x SE of the paired difference, 1% of |best net value|).

## Scoring (eval/score.py)
**Decision.** The actions the agent issued through action tools during the case are re-simulated from the case state
exactly like the oracle candidates. From that run:
- `acceptable` (0/1);
- regret = best − agent net value;
- normalised regret = regret / (best − no-action), only when the best beats no-action.

**Evidence.** Computed from the server's observation log and the submitted report:
- `unsupported`: a claim has no observation_id, or one that was never issued in this case;
- `mismatch`: the claimed value differs from every number in the cited observation by more than 2% (absolute 0.01 for numbers below 1);
- `unverified_action`: an issued action that was never run through `run_whatif` before it was issued. Reported only for condition (a).

**Primary comparison.** Mean acceptable rate and mean regret per condition, over all 30 cases and per family.
Paired across conditions by case. Wilcoxon signed-rank on regret; two-sided; no multiplicity correction (exploratory).

## Conditions
- (a) all tools;
- (b) all tools except `run_whatif`;
- (c) fixed rules (eval/rule_agent.py), which use only read tools and do not call `run_whatif`.

## Reporting
Every case and every condition is reported, including failures and errors. The oracle table is computed once and saved
before any agent result exists. Changing scenarios, objective or thresholds after seeing agent results requires a new,
dated protocol.

## Amendment 1 (2026-11-20, before any oracle result or agent run)
A pipeline test was run before the oracle. It used a 6-candidate mini-oracle on degradation_0 and breakdown_0, was used
only to test the code, and its values are not reused. It exposed two design faults:

1. **Events landed on stopped stations.** Target stations were drawn without checking their state. In degradation_0
   (S5), quality_0 (S30) and breakdown_0 (S25) the station was already stopped at the decision time, so the event was
   wiped out when the repair finished. Fix: target stations are drawn among stations that are running and not degraded
   at the decision time; same seed, same weights otherwise.
2. **Breakdowns needed no decision.** The simulator automatically sent parts to a parallel sibling when a station
   stopped, so the breakdown family had nothing to decide. Fix: `auto_reroute = False`. Parts follow the dispatch
   weights, so a stopped station builds a queue until it is repaired or the operator changes the dispatch. The
   simulator validation is re-run with this setting.

3. **Dispatch weights were ignored at entry stations, and avoiding a station could only be permanent.** Parallel entry
   stations (S24|S25) did not follow the dispatch weights. Avoiding a station lasted the whole 12 h, so a repaired
   station stayed idle while its sibling was overloaded. The rule agent hit this on S35. Fixes:
   - entry stations follow the weights;
   - `set_dispatch` takes an optional `hours` after which the normal split returns;
   - `get_station` reports `back_in_min` for a stopped station (the technician's repair estimate);
   - the candidate list gains `avoid4h_<s>` (avoid a parallel station for 4 h) for each of the 27 parallel stations, 164 candidates in all.

Scenarios are regenerated with these fixes. Everything else in this protocol is unchanged, including the rush target
rule. A known limitation of that rule: the target (1.04 x the no-action expectation) can exceed what arrivals allow in
12 h, so in rush cases every policy may pay some shortfall. Those cases measure how close an agent gets.
