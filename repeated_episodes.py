"""Repeated target episodes without resetting the monitor between episodes
(minor-revision round, Reviewer 3 point 4).

A healthy, synchronized system is monitored continuously. Targets arrive as
episodes of fixed dwell L slots; after an episode ends, the next one starts
at each H0 slot with probability 1/G (mean gap G). The bank is reset only at
an alarm (which is counted); no repair phase, since the question is the
spurious-alarm rate alone. Reported: the effective ARL (slots per alarm) of
the deployed H0-designed bank and of the hypothesis-mixture bank, each at its
own threshold for ARL = 1e4 under sustained H0.

Usage: python3 repeated_episodes.py <2LD|3LD>
Writes results/episodes_<sys>.json.
"""
import json, sys as _s
import numpy as np
from variant_calibration import SYSTEMS, p_eps, build

nm = _s.argv[1]
sys, fn, _ = SYSTEMS[nm]
p = p_eps(sys)
banks, _ = build(sys, p)
h_dep = json.load(open(f"results/final_{nm}.json"))["h_star"]
h_mix = json.load(open(f"results/variants_{nm}_mixture.json"))["h"]
DWELLS = (5, 10, 20)
GAPS = (100, 300, 1000, 3000)
N, T = 200, 10000                    # 2e6 slots per cell


def eff_arl(charts, R, h, L_dwell, gap, seed):
    rng = np.random.default_rng(seed)
    A = np.stack(charts)
    cdf0, cdf1 = np.cumsum(R["sync_H0"]), np.cumsum(R["sync_H1"])
    Z = np.zeros((A.shape[0], N))
    left = np.zeros(N, int)                       # slots left in the current episode
    alarms = 0
    for t in range(T):
        start = (left == 0) & (rng.random(N) < 1.0 / gap)
        left[start] = L_dwell
        intgt = left > 0
        u = rng.random(N)
        x = np.where(intgt, np.searchsorted(cdf1, u), np.searchsorted(cdf0, u))
        x = np.minimum(x, A.shape[1] - 1)
        Z = np.maximum(0.0, Z + A[:, x])
        hit = Z.max(axis=0) >= h
        alarms += int(hit.sum()); Z[:, hit] = 0.0
        left[intgt] -= 1
    return (N * T) / max(alarms, 1), alarms


out = dict(h_deployed=h_dep, h_mixture=h_mix, slots_per_cell=N * T, dwells=DWELLS, gaps=GAPS, cells={})
for name in ("deployed", "mixture"):
    charts, R, modes = banks[name]
    h = h_dep if name == "deployed" else h_mix
    for L_dwell in DWELLS:
        for gap in GAPS:
            arl, n = eff_arl(charts, R, h, L_dwell, gap, seed=7 + L_dwell + gap)
            out["cells"][f"{name}_L{L_dwell}_G{gap}"] = dict(eff_arl=arl, alarms=n)
            print(f"{nm} {name:8s} dwell={L_dwell:2d} gap={gap:5d}: effective ARL = {arl:9.0f}  ({n} alarms in {N*T:.0e} slots)", flush=True)
json.dump(out, open(f"results/episodes_{nm}.json", "w"), indent=1)
