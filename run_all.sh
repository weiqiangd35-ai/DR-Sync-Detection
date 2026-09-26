#!/usr/bin/env bash
# Reproduce every number, table and figure of the letter.
# Stages of pipeline.py are checkpointed in results/final_{2LD,3LD}.json.
set -e
python3 operating_points.py          # C, W*, W', C*, C*_eps, sigma_AB   (Sec. VI, R0)
python3 mismatch_table.py            # Table II                           (S3)
python3 coverage_s2.py               # single-LD chart coverage          (S2)
for sys in 2LD 3LD; do
  python3 pipeline.py $sys A         # thresholds h*, h_mix, h_u, h_a    (~2 min each)
  python3 pipeline.py $sys B         # Tables I, III, IV; Remark 1       (~3 min each)
  python3 pipeline.py $sys C         # identification; renewal (1e7 slots)
done
python3 fig2_roc.py                  # Fig. 2
python3 fig3_trajectory.py           # Fig. 3 (needs results/final_2LD.json)
