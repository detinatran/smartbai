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
