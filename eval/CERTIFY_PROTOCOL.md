# Protocol: twin-certified selective decisions (fixed 2026-11-20, before any code for this method is run)

## Question
An agent proposes actions for a production line and tests them in a digital twin. The twin is imperfect: its
parameters and its hidden state differ from the real line (the *plant*). Can a gate calibrated on past cases bound the
rate of **harmful** actions on the plant at a level α? A harmful action is one that does worse than doing nothing.
How much value does such a gate keep?

## Method under test: Certify
For each case:
1. **Propose.** A proposer inspects the line through the read tools and returns up to K = 5 candidate action lists, in
   the action grammar of `eval/common.py`. It has no `run_whatif`.
2. **Test in the twin.** Sequential paired comparison with no action as the standard, using common random numbers
   (CRN).
   - Replication r uses the same random stream for every candidate, and the same sampled hidden state.
   - Start: n0 = 4 replications of every candidate and of no action.
   - Each round adds 2 replications to every surviving candidate and to no action.
   - Elimination: candidate j is eliminated when the upper 95% bound of its mean paired difference D_j to no action is
     below the largest lower 95% bound among the candidates (t intervals on the paired differences).
   - Stop when one candidate is left or the budget of B = 64 twin replications is used (no-action replications count).
3. **Score.** j* is the surviving candidate with the largest mean D. Its score is s = mean(D_j*) − t(0.95, n−1) · SE.
4. **Gate.** Execute j* on the plant if s > τ. Otherwise abstain: no action, and the case goes to the manager with the
   shortlist.
5. **Calibrate.** τ is chosen by Learn-then-Test on the calibration split.
   - The thresholds form a fixed grid from +3000 down to −1000 net-value units, in steps of 50.
   - They are tested in that order, most conservative first.
   - For threshold τ_k the null hypothesis is H_k: P(harm | s > τ_k) > α.
   - The p-value is exact binomial: p_k = BinomCDF(K_k; N_k, α), where N_k is the number of calibration cases with
     s > τ_k and K_k the number of those that are harmful. p_k = 1 when N_k = 0.
   - Testing stops at the first p_k > δ. τ is the most permissive threshold certified before that point.
   - If no threshold is certified, the method always abstains, and that is reported.
   - α = 0.10, δ = 0.10.

Guarantee targeted: test cases are i.i.d. with the calibration cases (same generator, other seeds) and use the same
proposer. With probability ≥ 1 − δ over the calibration split, P(harm on plant | executed) ≤ α.

## Plant and twin
- **Plant.** `LineSim` with physical parameters θ* drawn per case. Each of the following is the `ASSUMED` value times an
  independent log-uniform factor in [1/1.5, 1.5]: `mtbf_h`, `mttr_min`, `temp_k`, `hall_coupling`, `precursor_h`,
  `degraded_defect_mult`, `maint_min`, `service_cv`, `hvac_kw_per_deg`, `cooling_high_coupling`. Prices and costs, the
  arrival rate and the line structure are known exactly.
- **Twin.** `LineSim` with the nominal `ASSUMED` parameters. It is synchronised from **plant observables only**:
  - station up/down, queues, speeds, dispatch weights, cooling level and scheduled maintenance;
  - remaining repair time, taken from the technician estimate `back_in_min` = true remaining × lognormal(0, 0.3);
  - outdoor temperature over the horizon, from a forecast = plant outdoor temperature + one bias drawn per case from N(0, 1.5²) °C;
  - the rush target, from the task text.
- **Hidden state is sampled, never copied.** For each twin replication:
  - station i is degraded with probability p_i, the `predict_failure` score;
  - its defect multiplier is (k_i + 1) / (E_i + 1), from the last 4 h of completed parts that visited it (k_i defects
    observed, E_i expected at the nominal rate);
  - failure clocks are fresh.
- **Harm label.**
  - Run the executed action and no action on the plant, from the case state, 12 h, 32 CRN replications, with streams
    disjoint from the twin's.
  - Δ_plant is the mean paired difference in net value.
  - harm = 1 if Δ_plant < 0.
  - The plant gain G of a case is Δ_plant if the action was executed, 0 if the method abstained.

## Cases (eval/scenarios_certify.py)
Six families:
- the five of `EVAL_PROTOCOL.md`;
- **quiet**: no injected event, where doing nothing is often right.

Magnitudes are drawn per case:
- repair 120–360 min;
- defect multiplier U[3, 8];
- heat +U[3, 8] °C for 12 h;
- rush target U[1.02, 1.08] × the plant no-action expectation.

Start times are drawn from 2025-05-01 to 2025-09-30, on the hour, between 06:00 and 18:00. Warm-up is 24 h on the plant.
Target stations are drawn as in amendment 1 (running and not degraded).

| Split | Cases | Seeds | Use |
|---|---|---|---|
| dev | 36 (6 per family) | 20000+ | debugging code and the proposer prompt only |
| calibration | 210 (35 per family) | 30000+ | choosing τ for each proposer; nothing else |
| test | 210 (35 per family) | 50000+ | all reported results; run once |

## Proposers
- **LLM.** An LLM agent through the tool server, with read tools only.
  - The model, prompt and decoding settings are named and frozen in `eval/certify/FROZEN.md` before the calibration
    split is run.
  - Only the dev split may be used to develop them.
- **Rule.** The fixed-rule agent's action list, plus each of its single actions, up to K.
- **Random-5.** 5 candidates drawn uniformly from the 163 non-null candidates of `eval/common.py`, seeded per case.

## Conditions (test split)
1. No action.
2. Rules: the rule agent executes its actions, with no twin and no gate.
3. LLM-ungated: the LLM proposer and twin testing; j* is always executed (τ = −∞).
4. LLM-naive: j* is executed if s > 0, the per-case twin bound with no calibration.
5. **LLM-Certify** (proposed): τ from LTT.
6. Rule-Certify.
7. Random5-Certify.

Each of conditions 5–7 has its own τ, calibrated on the calibration split.

## Hypotheses (confirmatory, test split)
- **H1, validity (falsification check).** Harm rate among executed decisions of LLM-Certify ≤ α.
  - Test: exact binomial test of H0: R ≤ 0.10 against R > 0.10, one-sided, level 0.05.
  - The guarantee is refuted if H0 is rejected.
  - The Clopper–Pearson 95% upper bound is reported.
- **H2, the per-case twin bound is not enough.** Harm rate among executed decisions of LLM-naive > α.
  - Test: exact binomial test, H0: R ≤ 0.10, one-sided.
- **H3, the gate keeps value.** The plant gain G of LLM-Certify is greater than 0.
  - Test: Wilcoxon signed-rank on G, one-sided; zero_method = "wilcox", so abstentions are dropped.
- **H4, proposals matter.** G of LLM-Certify is greater than G of Random5-Certify, paired by case.
  - Test: Wilcoxon signed-rank, one-sided.

H2–H4 form one family under Holm correction at 0.05. H1 is a falsification check tested on its own at 0.05. If the LLM
proposer cannot be run, H1–H4 are reported for Rule-Certify in its place, labelled as such, and H4 is dropped.

## Secondary and exploratory (reported, not tested confirmatorily)
- For every condition:
  - coverage (fraction executed);
  - harm rate with Clopper–Pearson interval;
  - mean G with a bootstrap 95% interval;
  - twin replications used;
  - LLM calls and tokens.
- Risk–coverage curves of conditions 3–7 on the test split, sweeping τ.
- Harm rate at matched coverage: LLM-naive and LLM-ungated re-thresholded to LLM-Certify's coverage.
- Results per family.
- Sensitivity to the size of the twin–plant gap: the plant factor range 1.0 (perfect parameters, hidden state still
  sampled) and 1.25, each with its own calibration. These need extra calibration runs and are exploratory.
- An online conformal-decision baseline (Lekeufack et al.): τ adapted along the test sequence towards a long-run harm
  rate of α.
- Plant oracle on the test split: all 164 candidates, 8 replications, on the plant. It gives regret against the best
  plant action, if compute allows.

## Fixed now, changed only by a dated amendment before the calibration split is run
K, n0, the round size, B, the elimination rule, the score, the τ grid, α, δ, the plant factor range, the forecast and
repair-estimate noise, the hidden-state sampling, the harm label, the plant replications, the split sizes and seeds,
and the hypotheses with their tests.

The dev split may reveal faults. Any fix is written as an amendment, with its reason, before the calibration split is
run. The test split is run once, after τ is fixed for every condition. Every case is reported, including errors.

## Implementation (to be written)
- `sim/twin_sync.py`: builds the twin from plant observables, with hidden-state sampling.
- `sim/line_sim.py`:
  - per-case plant parameters;
  - a noisy `back_in_min`;
  - an outdoor-temperature override for the forecast.
- `eval/scenarios_certify.py`: the three splits.
- `eval/certify/`:
  - `rs.py`: sequential paired testing against no action;
  - `ltt.py`: the gate;
  - `proposers.py`;
  - `label.py`: plant harm labels;
  - `run.py`, `summarize.py`.

Estimated compute on 12 CPU workers:
- twin testing: about 70 s per case per proposer;
- harm labels: about 35 s per distinct executed action;
- all splits: about 4 h, without the optional plant oracle (about 7 h).

## Amendment 1 (2026-11-20, before the calibration split is run)
Where these faults were found:
- a test of the twin on three cases of `EVAL_PROTOCOL.md` (breakdown_3, quality_3, degradation_0);
- reading the code.

No case of this protocol had been run when they were found.

1. **Degradation probability.** The `predict_failure` score is not a calibrated probability: many running stations
   scored 0.96–0.98. A twin sampling degradation from it overpredicted breakdowns, by 1,300–2,600 net-value units
   below the plant under no action. Fix: p_i is the output of a logistic model P(degraded now | the two `predict_failure`
   features).
   - It is fitted by `sim/fit_failure_model.py` on simulations with the nominal parameters, never on plant cases
     (seeds 900000+, 24 runs × 3 days, May–September 2025).
   - Binned calibration: mean predicted 0.50 against 0.51 observed; 0.93 against 0.89.
   - The `predict_failure` tool shown to agents is unchanged.
2. **Defect-drift estimator.** The formula (k_i + 1)/(E_i + 1) mixes part-level and station-level rates: a part that
   visited a station carries the defects of its whole route. In one case a z-score version attributed a ×136 drift to a
   station with few parts. Fix (`sim/twin_sync.py`, `estimate_drift`):
   - take the station with the smallest Poisson tail p-value P(Poisson(E_i) ≥ k_i);
   - if that p-value is below 0.01/39, attribute the excess to it: drift_i = 1 + (k_i − E_i)/(n_i · nominal p_i);
   - update the expectations and repeat, at most 3 times.
3. **Events left no trace at the decision time.** Degradation and defect drift were injected at the decision time.
   Sensors and quality data could not show them yet, so no method using observables could detect them. Fix:
   - degradation is injected 90 min before the decision time, and defect drift 180 min before;
   - the target station is drawn among stations running and not degraded at injection time;
   - the station may break down before the decision time, and such cases are kept.
4. **LTT screening.** As written, the most conservative thresholds select no case (p_k = 1), so testing would stop at
   once. Fix, following Messoudi & Mansar (arXiv 2610.19811):
   - thresholds selecting fewer than n_min = ⌈log δ / log(1 − α)⌉ = 22 calibration cases are skipped;
   - a skipped threshold is neither tested nor a reason to stop.

   Screening depends on the scores only, not on the harm labels.
5. **Details fixed in code.**
   - The twin uses the nominal per-station defect probabilities.
   - Plant cooling "normal" returns to the plant's own hall coupling. Before this fix it returned to the nominal value,
     which was a simulator bug; with the nominal plant it has no effect.
   - Random streams:
     - twin replication r samples its hidden state from [case seed, 7, r] and runs on [case seed, 7, r, 1], for every
       proposer;
     - plant labels use replications 1,000,000 + r of the plant's own streams, disjoint from the oracle (0–7) and from
       the twin.

The rest of the protocol is unchanged.

## Amendment 2 (2026-11-20, after the dev split, before the calibration split is run)
The dev split (36 cases, rule and random-5 proposers) showed that the harm label was mostly noise.
- With 32 plant replications, the standard error of Δ_plant had a median of about 330 net-value units.
- An action with no effect was therefore labelled harmful about half of the time, so the risk the gate had to control
  was mostly measurement noise.

Changes:
- **Harm label.** harm = 1 if Δ_plant < −m, where m = 1% of the case's plant no-action mean net value over 12 h
  (about 380 units). This is a non-inferiority margin, so the claim becomes P(material harm | executed) ≤ α.
- **Plant replications** rise from 32 to 96 (standard error about 190). An action with no effect is then mislabelled
  with probability about 2%.

The method is not changed: K, n0, the round size, B, the elimination rule, the score, the τ grid, α and δ. The dev
results are not used for anything but this fix and are not reported as results.

## Result of the protocol as amended up to amendment 2 (v1, 2026-11-21)
Calibration and test splits were run once each (rule and random-5 proposers; no LLM proposals).
- LTT certified no threshold for either proposer: on calibration, the most conservative testable threshold selected
  24 cases with 7 (rule) and 5 (random-5) harms. Both Certify conditions therefore abstained on all 210 test cases.
- H1: not refuted, trivially (0 executed). H2: 7/48 harmful, p = 0.20, Holm 0.40, not supported. H3: G = 0, not
  supported. H4: dropped (no LLM proposer).
- On the test split the twin score was not informative: for the rule proposer the harm rate among executed actions
  was 0.19 at coverage 0.15 (τ = 250) and 0.08 at full coverage.
This result is reported as it is. Amendment 3 is a new run on fresh calibration and test seeds (v2), not a re-analysis.

## Amendment 3 (2026-11-21, after the v1 result, before any v2 calibration or test case is run)
Diagnosis on the dev seeds only (20000+: the 6 dev cases per family and 30 extra rush and 30 extra quiet cases drawn
with the same generator; `eval/certify/diag_twin.py`):
- **Common random numbers did not work.** The simulator drew arrivals, routing, service times, defects, failure clocks,
  repairs and sensor noise from one random stream. Any action shifted every later draw, so paired differences were
  nearly as noisy as unpaired ones, both in the twin and in the plant labels.
- **The rush family amplified this.** While a rush order is short, a part is worth its value plus the shortfall
  penalty (10 + 40 units), so throughput noise is multiplied about five times. With 96 plant replications the median
  SE of the rush harm label was about 1,180 units against a margin of about 380, so an action with no effect was
  labelled harmful about 36% of the time (other families: 1–10%). In the v1 calibration split, 16 of 35 rush cases
  were labelled harmful for each proposer.
- With one stream per purpose and station, the twin's mean gain on dev rush cases matched the plant's
  (speed_trunk_115: twin +1,257, plant +1,370; avoid_S16: −11,162 vs −11,554), and the median paired SE fell 2–5 times.
  The per-case correlation of twin and plant gains stays low (0.1–0.35 for most probes): the plant's failure clocks
  are hidden and the twin samples them, which the twin SE reflects.

Changes:
1. **Random streams** (`sim/line_sim.py`, `rng_streams = "split"`): arrivals, entry choice, and per station routing,
   service, defects, failure clocks and repairs, plus sensor noise, each have their own stream derived from the
   replication seed. The twin and the plant use the same simulator. An action that does not change a station's draws
   leaves them aligned; an inert action gives exactly the no-action result.
2. **Plant label replications** (`eval/certify/run.py`): paired plant replications start at 96 and grow in batches of
   96, with the same batches for no action, until SE(Δ_plant) ≤ m/2 or 768 replications. The margin m is fixed from
   the first 96 no-action replications. An action with no effect is then mislabelled with probability about 2%
   whenever the cap is not reached; the number of replications is reported per case.
3. **Degradation model** refitted with the new simulator by the same procedure (`sim/fit_failure_model.py`, seeds
   900000+, nominal parameters).
4. **Fresh splits.** Calibration seeds 60000+ and test seeds 70000+ (35 cases per family each, same generator).
   The dev split (20000+) is unchanged and is used only to check that the code runs.

Unchanged: the method (K, n0, round size, B, elimination rule, score, τ grid, α = δ = 0.10, LTT screening), the plant
factor range, the observation noise, the hidden-state sampling, the harm definition and margin, the conditions, and
the hypotheses with their tests. v2 is run once: calibration, then test.

## Result of the protocol as amended up to amendment 3 (v2, 2026-11-22)
Dev, calibration and test splits were run once each (rule and random-5 proposers; no LLM proposals). No case failed
after the retry rule of `run.py`. Plant labels: median 96 replications (max 768); median SE(Δ_plant)/m 0.08 (rule) and
0.11 (random-5) on the calibration split.
- **Calibration:** τ = None for both proposers. Rule: the first threshold with at least n_min executed cases was
  τ = 400 (22 cases, 1 harm, p = 0.34 > δ), so fixed-sequence testing stopped there. Random-5: τ = 0 (38 cases,
  2 harms, p = 0.25). Harm rate if every tested j* is executed: rule 0.034, random-5 0.081.
- **Test (confirmatory, Rule-Certify in place of LLM-Certify):** Certify abstained on 210 of 210 cases.
  H1: not refuted (no executed decision). H2: rule-naive harm 3/128 = 0.023 (CP 95% 0.005–0.067), p = 0.9998, Holm 1,
  not supported; the per-case twin bound alone kept the harm rate below α. H3: G = 0, not supported. H4: dropped.
- **Secondary (test):** rules executed directly: coverage 0.99, harm 16/207 = 0.077, mean G 468 (bootstrap 95%
  143–795). Rule-ungated: harm 10/207 = 0.048, mean G 494. Rule-naive: coverage 0.61, harm 0.023, mean G 421.
  Random5-ungated: harm 28/210 = 0.133, mean G 137 (−108–348). Random5-naive: coverage 0.23, harm 0.082, mean G 73.
  Risk is still highest among the cases with the largest scores (rule: harm 0.08 above τ = 500, 0.02 above τ = 0;
  random-5: 0.29–0.33 above τ ≥ 250), and roughly flat at 0.02–0.04 (rule) from τ = 0 downwards.
- **Reading:** amendment 3 repaired the twin and the labels; the remaining failure is the test order of the gate.
  Testing from the most conservative threshold first meets the high-score cases, where risk is highest and the
  threshold sets are smallest, before any well-populated threshold, and a single harm among the first 22 cases stops
  the sequence.
- **Exploratory (not confirmatory; the order was chosen after seeing this result):** with the order of amendment 4
  (from τ = 0 downwards), the v2 calibration split certifies τ = −1000 for rule (191 cases, 2 harms, p = 4·10⁻⁷) and
  none for random-5. On the v2 test split Rule-Certify would then execute 193 of 210 cases with 7 harms (0.036, CP 95%
  upper 0.073) and mean G 576 (Wilcoxon p = 4·10⁻²²). `summarize.py --ltt a2` reproduces the confirmatory v2 result;
  `--ltt a4` gives this exploratory one.

## Amendment 4 (2026-11-22, after the v2 result, before any v4 calibration or test case is run)
Motivation: the v2 result above. The gate's fixed-sequence test order started at τ = 3000, among the highest twin
scores, where the plant risk is highest (winner's curse of the selection over K candidates) and few cases are
selected. The order was data-independent and valid, but it could not certify any threshold once one of the first
well-populated thresholds held a single harm.

Changes:
1. **LTT test order** (`eval/certify/ltt.py`, `GRID_A4`): fixed-sequence testing over τ = 0, −50, …, −1000, in this
   order, with the same exact binomial p-values, the same screening of thresholds with fewer than n_min = 22 executed
   cases, and the same stopping rule; τ is the last certified threshold, or None (always abstain). τ = 0 is the
   per-case twin bound (LLM-naive / Rule-naive), so Certify tests that bound first and only relaxes it while the
   calibration harm rate stays certifiably below α. The order is fixed before any v4 case is run and does not depend
   on data; family-wise error control at δ is as in amendment 1. Thresholds above 0 are no longer tested.
2. **Fresh splits** (`eval/scenarios_certify.py --version v4`): calibration seeds 80000+ and test seeds 90000+
   (35 cases per family each, same generator). The dev split (20000+) is unchanged and is not run again.

Unchanged: the simulator, twin and random streams of amendment 3, the degradation model fitted for v2 (reused as is),
the plant label rule and margin, K, n0, round size, B, the elimination rule and score, α = δ = 0.10, n_min, the
conditions, and the hypotheses with their tests and Holm family. Rule-Certify stands in for LLM-Certify and H4 is
dropped unless LLM proposals frozen by `FROZEN.md` exist before the v4 calibration split is run. v4 is run once:
calibration, then test; the v4 result is reported whatever it is, alongside v1 and v2.
