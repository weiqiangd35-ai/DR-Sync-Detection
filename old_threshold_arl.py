"""Bank-level ARL at the threshold used by the previous version.

The rejected version set h = ln 500 (Wald approximation for a single chart)
and reported ARL = 500. This script measures what the *bank* ARL actually was
at that threshold, at the previous design C and at the deployed design C_eps.
The numbers are quoted in the response letter (R2.1); they appear nowhere in
the manuscript.

Usage: python3 old_threshold_arl.py        (about 30 s)
"""
import numpy as np
from systems import SYS2, SYS3
from fastsim import first_passage, mean_se

EPS = 0.01
H_OLD = float(np.log(500.0))


def bank_arl(sys, p, h, N=5000, seed=99):
    f0 = sys.f_sync(p)
    M = sys.modes(p)
    keys = list(f0.keys())
    charts = [np.array([np.log(M[m][k] / f0[k]) if M[m][k] > 0 else -50.0
                        for k in keys]) for m in M]
    P0 = np.array([f0[k] for k in keys])
    T, _ = first_passage(charts, P0, [h], N=N, seed=seed, maxT=400000)
    return mean_se(T[0])


if __name__ == "__main__":
    print(f"bank ARL at the previous threshold h = ln 500 = {H_OLD:.3f}")
    for nm, sys in (("2-LD", SYS2), ("3-LD", SYS3)):
        p_eps = (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])
        for label, p in (("C    ", sys.p), ("C_eps", p_eps)):
            m, se, n = bank_arl(sys, p, H_OLD)
            print(f"  {nm} at {label} (p = {p:.4f}): {m:7.0f} +/- {se:.0f}   "
                  f"({n} paths)")
    print("reference values (seed 99): 2-LD 4206 / 4561; 3-LD 8282 / 9097")
