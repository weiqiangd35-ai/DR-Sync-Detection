"""Supplementary S2: coverage of the single-LD charts.
The P({k}) chart's increment depends on (s, u_k) only, so its drift is the
same under every mode that decouples LD k (and negative when LD k stays
synchronized). More generally, if u_k is independent of s with ANY marginal
pi', the drift is  sum_s Pr(s) D(pi'||Pr(.|gamma_k^s)) - D(pi'||pibar) >= 0.
Computed at the deployed eps-design, as in the letter."""
import numpy as np
from systems import SYS2, SYS3, kl

EPS = 0.01

def bern_kl(a, b):
    return sum(x * np.log(x / y) for x, y in ((a, b), (1 - a, 1 - b)) if x > 0)

for nm, sys, k in (("3-LD", SYS3, 2), ("2-LD", SYS2, 1)):
    p = (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])
    f0 = sys.f_sync(p); M = sys.modes(p); keys = list(f0)
    chart = f"P{{LD{k+1}}}"
    L = {x: np.log(M[chart][x] / f0[x]) for x in keys}
    print(f"{nm}: chart {chart} at p_eps = {p:.4f}; own I(H0) = {kl(M[chart], f0):.4f}")
    for m in M:
        drift = sum(M[m][x] * L[x] for x in keys)
        tag = "decouples LD%d" % (k + 1) if f"LD{k+1}" in m or m in ("G", "F") else "LD%d synchronized" % (k + 1)
        print(f"   drift under {m:<12} {drift:+.4f}   ({tag})")
    pA, pB = sys.locA[0], sys.locB[0]           # local P_f under rule A / B
    pibar = p * pA + (1 - p) * pB
    drift = lambda q: p * bern_kl(q, pA) + (1 - p) * bern_kl(q, pB) - bern_kl(q, pibar)
    qs = np.linspace(1e-6, 1 - 1e-6, 20001); d = np.array([drift(q) for q in qs])
    print(f"   independent coin, marginal pi' = P(u_k = 1):")
    for lab, q in (("private coin (pibar)", pibar), ("stuck on rule A", pA),
                   ("stuck on rule B", pB), ("always 1", 1 - 1e-12)):
        print(f"      {lab:<22} {drift(q):+.4f}")
    print(f"      minimum over pi'       {d.min():+.4f}  (at pi' = {qs[d.argmin()]:.3f})\n")
