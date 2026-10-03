#!/usr/bin/env bash
# Reproduce every number, table and figure of the letter.
# Stages of pipeline.py are checkpointed in results/final_{2LD,3LD}.json.
set -e
python3 operating_points.py          # C, W*, W', C*, C*_eps, sigma_AB   (Sec. VI, R0)
python3 mismatch_table.py            # Table II                           (S3)
python3 coverage_s2.py               # single-LD chart coverage          (S2)
python3 old_threshold_arl.py         # bank ARL at the old h = ln 500    (response letter R2.1)
python3 h1_contamination.py          # sustained targets, variants, slip (Sec. VI, R3)
for s in 3LD 2LD; do for b in deployed gated mixture; do for st in cal check meas; do
  python3 variant_calibration.py $s $b $st   # variants at their own ARL (R3, Table V)
done; done; done
python3 variant_calibration.py report
for sys in 2LD 3LD; do
  python3 pipeline.py $sys A         # thresholds h*, h_mix, h_u, h_a    (~2 min each)
  python3 pipeline.py $sys B         # Tables I, III, IV; Remark 1       (~3 min each)
  python3 pipeline.py $sys C         # identification; renewal (1e7 slots)
done
python3 fig2_roc.py                  # Fig. 2
python3 fig3_trajectory.py           # Fig. 3 (needs results/final_2LD.json)
