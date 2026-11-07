# Detecting Loss of Synchronization in Parallel Decision Fusion with Dependent Randomization

Code that reproduces every number, table and figure in the letter by
W. Dong and J. Zhou (submitted to *IEEE Signal Processing Letters*).

The benchmark systems are those of W. Dong and M. Kam, "Dependent
randomization in parallel binary decision fusion," *IEEE/CAA Journal of
Automatica Sinica*, vol. 8, no. 2, pp. 361–376, 2021.

## Requirements

Python 3.10 or later with `numpy` and `matplotlib` (`pip install -r requirements.txt`).
Tested with Python 3.12, numpy 2.4, matplotlib 3.10.

## Quick start

```bash
./run_all.sh
```

All simulations use fixed seeds, so the outputs are reproduced exactly.
The full run takes roughly 25–30 minutes on a laptop; `results/` already
contains the reference outputs, so each stage can also be run on its own.

## What each script reproduces

| Script | Output | Where it appears in the letter |
|---|---|---|
| `operating_points.py` | Operating points C, W\*, W′, C\* of the parent paper; recalibrated stopgap C\*_ε; chord slope σ_AB. Each value is checked against the letter and marked MATCH. | Sec. VI; Information for Reproducibility (R0) |
| `mismatch_table.py` | Post-correction P_f for every (true, assumed) failure mode; worst-case P_d loss | Table II; Supplementary S3 |
| `coverage_s2.py` | Drift of the single-LD chart under every failure mode and under arbitrary independent local behavior | Supplementary S2 |
| `pipeline.py <2LD\|3LD> A` | Thresholds: MC-calibrated h\* (bank), h_mix (mixture), h_u (decision-stream only), analytic h_a; each verified by an independent run | Table I caption; R0; R1 |
| `pipeline.py <2LD\|3LD> B` | Signatures I(H₀), I(H₁), false-alarm excess Δ, delay bound, measured delay, critical rate 1/λ\*; bound versus measured delay over h; mixture and decision-stream-only comparisons; group-mode chart under a single-LD failure | Tables I, III, IV; Remark 1; Sec. VI; R1, R2 |
| `pipeline.py <2LD\|3LD> C` | Mode-identification accuracy at the alarm; long-run P_f of the idealized zero-start renewal model (`fastsim.renewal`: bank started at each failure onset, inactive while synchronized, instantaneous repair, no spurious alarms; 10⁷ slots) | Sec. VI; S3; R1 |
| `old_threshold_arl.py` | Bank-level ARL at the previous version's threshold h = ln 500, at C and at C_ε (quoted only in the response letter, R2.1) | — |
| `h1_contamination.py` | Sensitivity to sustained targets, the gated and mixture variants, clock slip (Sec. VI "Sustained targets", R3, Table V) | — |
| `variant_calibration.py` | The two variants as banks: threshold at bank ARL 10⁴, episode false alarms, delays with/without a coincident target, bound and critical rate; mixture weight interval; `report` also checks E_f0[exp(increment)] ≤ 1 per chart (R1's analytic bound) (R3, Table V). Staged: `<sys> <bank> <cal|check|meas>`, then `report` | `results/variants_<sys>_<bank>.json` |
| `a4_scan.py` | Safe-correction assumption (A4): smallest backoff ε for which every (true, assumed) entry of Table II stays at or below α, with the binding pairs and the values at ε = 0.001; P_f and P_d cost of a healthy system under the stopgap keyed to each m̂ (row G of Table II); 1/λ\* per mode for the bank and the failure-mode mixture (bound-based and from measured delays) and the all-mode maxima | Sec. II (A4); Sec. V; S3; R2 |
| `continuous_loop.py <2LD\|3LD> [rho]` | Continuously operating monitor with verification/repair: the bank runs at every healthy or failed slot, failures arrive during healthy operation, every alarm (true or spurious) is followed by `rho` repair slots under the stopgap keyed to the alarming chart's m̂, then re-synchronization and a bank restart; long-run P_f at λ\*, λ\*_ex, 2λ\*_ex and with modes drawn uniformly from M; cycle mean delay; spurious-alarm rate (10⁷ slots per configuration). Run with `rho` = 100 and 0 | Sec. V (renewal model), Sec. VI, R1; `results/continuous_<sys>_rho<rho>.json` |
| `repeated_episodes.py <2LD\|3LD>` | Repeated target episodes (dwell 5/10/20 slots, mean gaps 100/300/1000/3000) on a healthy system with no reset between episodes: effective ARL of the deployed bank and of the hypothesis-mixture bank at their own thresholds (2×10⁶ slots per cell) | Sec. VI ("Sustained targets"), R3; `results/episodes_<sys>.json` |
| `s4_unequal.py` | S4 example with unequal local designs: the 2-LD benchmark with LD2's rules moved to (0.30, 0.84)/(0.18, 0.70), same AND/OR fusion, α and ε; recalibrates the four-chart bank at ARL 10⁴ and prints I_m(H₀), Δ_m, ℓ_max, the bound (4) and 1/λ\*_m per mode. Runs the equal-design benchmark first and reproduces Table I (h\* = 6.99; 426/312/397) | S4; `results/s4_unequal_2LD.json` |
| `fig2_roc.py` | Team ROC, deterministic points and the operating point after an undetected failure (3-LD) | Fig. 2 |
| `fig3_trajectory.py` | One realization of the group-mode CUSUM at the calibrated threshold | Fig. 3 |

Fig. 1 (system architecture) is a drawing, not generated by code.

## Modules

- `systems.py` — the two benchmark systems and the joint laws of the
  fusion-center record (s_t, U_t) under synchronization and under each
  failure mode (group mode G, partial modes P(·), full mode F).
- `fastsim.py` — vectorized CUSUM simulation. For a CUSUM started at zero
  the statistic does not reset before its first passage, so the first-passage
  times over *every* threshold below h_max are read off the same path; one run
  therefore yields the whole ARL(h) or delay(h) curve. Thresholds are found by
  interpolating ln ARL(h) to the target and verified with an independent seed.
- `trajectory_plot.py` — plotting helpers for Fig. 3.

## Conventions

- ARL is the mean first-passage time from zero under synchronization, for the
  *bank* T_bank = min_m T_m (not for an individual chart); target ARL = 10⁴.
- All closed-loop quantities are evaluated at the deployed ε-design C_ε
  (ε = 0.01), whose switching probability p_ε puts the synchronized point at
  P_f = α − ε.
- Delays are zero-start unless stated otherwise: the statistic is at 0 when
  the failure begins (the CUSUM's worst case). `continuous_loop.py` also
  reports the cycle mean delay of a continuously running bank, whose state at
  a failure onset is whatever healthy operation left it.
- `mismatch_table.py` evaluates the stopgap at `EPS = 0.01`; set `EPS` to
  another value to check the safe-correction condition (A4) of the letter at
  another backoff; `a4_scan.py` finds the smallest safe margins, 0.00173 in
  3-LD and 0.00807 in 2-LD (0.00172 and 0.00806 fail).

## Reference outputs

`results/final_2LD.json` and `results/final_3LD.json` hold every quantity
computed by `pipeline.py`, including standard errors;
`results/variants_<sys>_<bank>.json`, `results/continuous_<sys>_rho<rho>.json`,
`results/episodes_<sys>.json` and `results/s4_unequal_2LD.json` hold the
outputs of `variant_calibration.py`, `continuous_loop.py`,
`repeated_episodes.py` and `s4_unequal.py`.
