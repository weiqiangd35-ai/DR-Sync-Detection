"""Safe-correction assumption (A4), the false-alarm correction policy, and the
all-mode comparison of bank and mixture (minor-revision round; S3, Sec. V, R2).

  * scans the backoff margin eps and reports the smallest eps for which every
    (true mode, assumed mode) entry of Table II stays at or below alpha, with
    the binding pair and the values at eps = 0.001 (Reviewer 4's counterexample);
  * for a healthy system under the stopgap keyed to each mhat (the law of row G
    of Table II), prints Pf and the Pd cost relative to C_eps;
  * from results/final_<sys>.json, prints 1/lambda* per mode for the bank and
    the failure-mode mixture, bound-based and from measured delays, and the
    all-mode maxima.

Usage: python3 a4_scan.py          (no arguments; prints everything)
"""
import json
import math
import numpy as np
from systems import SYS2, SYS3
import mismatch_table as mt

EPS0 = 0.01


def stopgap(sys, eps):
    """q per assumed mode, Pf and Pd for every (true, assumed) pair, C_eps."""
    p = (sys.B[0] - (sys.alpha - eps)) / (sys.B[0] - sys.A[0])
    M0, M1 = sys.modes(p, 0), sys.modes(p, 1)
    tgt = sys.alpha - eps
    R = {m: mt.rule_points(sys, M0[m], M1[m]) for m in M0}
    q = {m: (R[m][1][0] - tgt) / (R[m][1][0] - R[m][0][0]) for m in M0}
    pf = {(m, mh): q[mh] * R[m][0][0] + (1 - q[mh]) * R[m][1][0] for m in M0 for mh in M0}
    pd = {(m, mh): q[mh] * R[m][0][1] + (1 - q[mh]) * R[m][1][1] for m in M0 for mh in M0}
    f0, f1 = sys.f_sync(p, 0), sys.f_sync(p, 1)
    pfC = sum(v for (cs, U), v in f0.items() if sys.fuse(cs, U))
    pdC = sum(v for (cs, U), v in f1.items() if sys.fuse(cs, U))
    return p, list(M0), q, pf, pd, (pfC, pdC)


def a4_threshold(sys):
    """smallest eps (bisection, 1e-8) with max over pairs of Pf <= alpha"""
    lo, hi = 1e-4, EPS0
    ok = lambda e: max(stopgap(sys, e)[3].values()) <= sys.alpha
    assert ok(hi) and not ok(lo)
    while hi - lo > 1e-8:
        mid = (lo + hi) / 2
        if ok(mid): hi = mid
        else: lo = mid
    _, modes, _, pf, _, _ = stopgap(sys, lo)
    worst = max(pf.values())
    pairs = [k for k, v in pf.items() if abs(v - worst) < 1e-9]
    return hi, pairs


if __name__ == "__main__":
    for nm, sys in (("3LD", SYS3), ("2LD", SYS2)):
        print(f"=== {nm} (alpha = {sys.alpha}) ===")
        thr, pairs = a4_threshold(sys)
        print(f"(A4) holds for eps >= {thr:.7f} (smallest safe margin, rounded up to three significant "
              f"digits: {math.ceil(thr * 1e5) / 1e5:.5f}); binding pairs (true, assumed): {pairs}")
        p, modes, q, pf, pd, C = stopgap(sys, 0.001)
        print(f"eps = 0.001: p_eps = {p:.6f}, q_F = {q['F']:.6f}, Pf(true G, assumed F) = {pf[('G','F')]:.6f}, "
              f"worst = {max(pf.values()):.6f} (> alpha: {max(pf.values()) > sys.alpha})")
        p, modes, q, pf, pd, C = stopgap(sys, EPS0)
        print(f"eps = {EPS0}: C_eps = ({C[0]:.4f}, {C[1]:.4f}); healthy system under the stopgap keyed to mhat (row G):")
        for mh in modes:
            print(f"   mhat = {mh:12s}  q = {q[mh]:.4f}  Pf = {pf[('G', mh)]:.4f}  Pd = {pd[('G', mh)]:.4f}  Pd cost = {C[1]-pd[('G', mh)]:.4f}")
        print(f"   Pf range {min(pf[('G',mh)] for mh in modes):.4f}-{max(pf[('G',mh)] for mh in modes):.4f}, "
              f"Pd cost range {min(C[1]-pd[('G',mh)] for mh in modes):.4f}-{max(C[1]-pd[('G',mh)] for mh in modes):.4f}")
        r = json.load(open(f"results/final_{nm}.json"))
        rows, mix = r["rows"], r["mix"]
        print("1/lambda* per mode (bank bound / bank measured | mixture bound / mixture measured):")
        for m in rows:
            D = rows[m]["Delta"]
            print(f"   {m:12s} {D*rows[m]['bnd']/EPS0:6.0f} / {D*rows[m]['dmc']/EPS0:6.0f} | "
                  f"{D*mix[m]['bnd']/EPS0:6.0f} / {D*mix[m]['dmc']/EPS0:6.0f}")
        mx = lambda key, src: max(rows[m]["Delta"] * src[m][key] / EPS0 for m in rows)
        print(f"all-mode maxima: bank {mx('bnd', rows):.0f} (bound) {mx('dmc', rows):.0f} (measured); "
              f"mixture {mx('bnd', mix):.0f} (bound) {mx('dmc', mix):.0f} (measured)\n")
