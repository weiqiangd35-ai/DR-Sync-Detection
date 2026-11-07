"""S4 example: a 2-LD system with unequal local designs.

LD1 keeps the benchmark rules of SYS2; LD2's two rules are moved to
(P_f, P_d) = (0.30, 0.84) under strategy A and (0.18, 0.70) under B, with the
same AND/OR fusion pair, alpha = 0.2009 and the eps-backoff design C_eps
(eps = 0.01). The symmetry reduction then fails: P({1}) and P({2}) are
distinct modes with their own signature, excess and critical rate.

Output: results/s4_unequal_2LD.json and a printed table (I_m(H0), Delta_m,
l_max, bound (4) at the calibrated bank threshold h*, and 1/lambda*_m).
Usage: python3 s4_unequal.py
"""
import json
import numpy as np
from itertools import product
from systems import Sys, SYS2, kl
from fastsim import calibrate, first_passage, mean_se

ARL, EPS = 1e4, 0.01


class SysU(Sys):
    """Sys with per-LD local rules: locA/locB are lists of (P_f, P_d) tuples."""

    def __init__(s, n, locA, locB, thrA, thrB, p, alpha, A=None, B=None):
        s.n, s.locA, s.locB, s.thrA, s.thrB = n, locA, locB, thrA, thrB
        s.p, s.alpha = p, alpha
        s.U = list(product([0, 1], repeat=n))
        # global points of the two strategies; the benchmark's published
        # (rounded) values can be passed in so that the equal-design check
        # reproduces Table I exactly
        s.A = A if A is not None else s.global_point(locA, thrA)
        s.B = B if B is not None else s.global_point(locB, thrB)

    def prod(s, ops, Uv, h=0):
        r = 1.0
        for k, u in enumerate(Uv):
            r *= s.bit(ops[k], u, h)
        return r

    def global_point(s, ops, thr):
        pt = []
        for h in (0, 1):
            pt.append(sum(s.prod(ops, Uv, h) for Uv in s.U if sum(Uv) >= thr))
        return tuple(pt)

    def f_part(s, yb, p=None, h=0):
        p = s.p if p is None else p
        law = {}
        for cs, ps, ops in ((0, p, s.locA), (1, 1 - p, s.locB)):
            for Uv in s.U:
                pr = ps
                for k, u in enumerate(Uv):
                    pr *= (p * s.bit(s.locA[k], u, h) + (1 - p) * s.bit(s.locB[k], u, h)) if k in yb \
                        else s.bit(ops[k], u, h)
                law[(cs, Uv)] = law.get((cs, Uv), 0) + pr
        return law


def analyze(sys, label, calib_range=(6.0, 8.0), N=5000, seed=3):
    p = (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])      # eps-backoff coin bias
    f0, M = sys.f_sync(p), sys.modes(p)
    f01, M1 = sys.f_sync(p, h=1), sys.modes(p, h=1)
    keys = list(f0.keys())
    P0 = np.array([f0[k] for k in keys])
    L = {m: np.array([np.log(M[m][k] / f0[k]) if M[m][k] > 0 else -50.0 for k in keys]) for m in M}
    fuse = np.array([1.0 if sys.fuse(k[0], k[1]) else 0.0 for k in keys])
    pf = lambda law: float(sum(law[k] * fuse[i] for i, k in enumerate(keys)))
    h_star, cal = calibrate([L[m] for m in M], P0, ARL, *calib_range, N=N, seed=seed)
    h_a = float(np.log(2 * len(M) * ARL))
    T, _ = first_passage([L[m] for m in M], P0, [h_star], N=N, seed=99, maxT=int(40 * ARL))
    arl_check = mean_se(T[0])
    rows = {}
    for m in M:
        I0, I1 = kl(M[m], f0), kl(M1[m], f01)
        D, lmax = pf(M[m]) - sys.alpha, float(L[m].max())
        bnd = (h_star + lmax) / I0
        rows[m] = dict(I0=I0, I1=I1, Delta=D, lmax=lmax, bnd=bnd,
                       inv_lstar=D * bnd / EPS if D > 0 else None)
    out = dict(label=label, locA=sys.locA, locB=sys.locB, A=sys.A, B=sys.B, alpha=sys.alpha,
               p_eps=p, h_star=h_star, h_a=h_a, arl_check=arl_check,
               calib_se_at_target=cal["se_at_target"], rows=rows)
    print(f"{label}: A={tuple(round(x,4) for x in sys.A)} B={tuple(round(x,4) for x in sys.B)} "
          f"p_eps={p:.4f} h*={h_star:.3f} (ARL check {arl_check[0]:.0f}+-{arl_check[1]:.0f}) h_a={h_a:.2f}")
    for m, r in rows.items():
        inv = f"{r['inv_lstar']:.0f}" if r["inv_lstar"] else "--"
        print(f"  {m:<8} I0={r['I0']:.4f} I1={r['I1']:.4f} Delta={r['Delta']:+.4f} "
              f"lmax={r['lmax']:.3f} bnd={r['bnd']:.0f} 1/lstar={inv}")
    return out


if __name__ == "__main__":
    equal = SysU(2, [SYS2.locA] * 2, [SYS2.locB] * 2, 2, 1, 0.5, SYS2.alpha, SYS2.A, SYS2.B)
    unequal = SysU(2, [SYS2.locA, (0.30, 0.84)], [SYS2.locB, (0.18, 0.70)], 2, 1, 0.5, SYS2.alpha)
    res = {"equal_check": analyze(equal, "2-LD benchmark (equal designs, consistency check)", (6.6, 7.4)),
           "unequal": analyze(unequal, "2-LD with LD2 rules (0.30,0.84)/(0.18,0.70)")}
    json.dump(res, open("results/s4_unequal_2LD.json", "w"), indent=1, default=float)
    print("wrote results/s4_unequal_2LD.json")
