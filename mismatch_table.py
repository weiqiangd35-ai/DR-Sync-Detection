"""Table II (Supplementary S3): post-correction Pf for every
(true mode, assumed mode) pair at the eps-design.  The stopgap picks the
DFC coin bias q from the assumed mode so that Pf = alpha - eps under that
mode's law; the table reports the Pf actually obtained under the true mode.
Also prints the worst-case Pd loss relative to the matched correction."""
import numpy as np
from systems import SYS2, SYS3

EPS = 0.01

def rule_points(sys, law0, law1):
    """(Pf, Pd) if the DFC always applied rule A, and if always rule B."""
    out = []
    for thr in (sys.thrA, sys.thrB):
        pf = sum(v for (cs, U), v in law0.items() if sum(U) >= thr)
        pd = sum(v for (cs, U), v in law1.items() if sum(U) >= thr)
        out.append((pf, pd))
    return out

def table(sys):
    p = (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])
    M0, M1 = sys.modes(p, 0), sys.modes(p, 1)
    tgt = sys.alpha - EPS
    R = {m: rule_points(sys, M0[m], M1[m]) for m in M0}
    q = {m: (R[m][1][0] - tgt) / (R[m][1][0] - R[m][0][0]) for m in M0}
    pf = {(m, mh): q[mh] * R[m][0][0] + (1 - q[mh]) * R[m][1][0] for m in M0 for mh in M0}
    pd = {(m, mh): q[mh] * R[m][0][1] + (1 - q[mh]) * R[m][1][1] for m in M0 for mh in M0}
    loss = max(pd[(m, m)] - pd[(m, mh)] for m in M0 for mh in M0)
    return list(M0), pf, loss

if __name__ == "__main__":
    for nm, sys in (("2-LD", SYS2), ("3-LD", SYS3)):
        modes, pf, loss = table(sys)
        print(f"{nm}  (alpha = {sys.alpha}; rows = true mode, columns = assumed mode)")
        print("  " + " " * 12 + "".join(f"{m:>12}" for m in modes))
        for m in modes:
            print(f"  {m:<12}" + "".join(f"{pf[(m, mh)]:12.4f}" for mh in modes))
        worst = max(pf.values())
        print(f"  worst Pf = {worst:.4f} (<= alpha: {worst <= sys.alpha}); "
              f"worst Pd loss vs matched = {loss:.4f}\n")
