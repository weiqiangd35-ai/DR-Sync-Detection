"""Reproduce the operating points of [7] used in the letter (Sec. VI, R0):
C (synchronized), W* (group mode G), W' (partial mode), C* (after the
corrective re-randomization of [7, eq. (25)]); plus the eps-design stopgap
C*_eps and the chord slope sigma_AB behind the 0.91*eps backoff cost.
Every value is checked against the number printed in the letter."""
import numpy as np
from systems import SYS2, SYS3

EPS = 0.01

def point(sys, law0, law1, rule_of_coin=None):
    """(Pf, Pd) of a joint law of (s, U); fusion rule indexed by the DFC coin."""
    pf = sum(v for (cs, U), v in law0.items() if sys.fuse(cs, U))
    pd = sum(v for (cs, U), v in law1.items() if sys.fuse(cs, U))
    return pf, pd

def exact_AB(sys):
    """Global points of the two deterministic strategies, from the local rules."""
    def pt(loc, thr, h):
        return sum(sys.prod(loc, U, h) for U in sys.U if sum(U) >= thr)
    A = (pt(sys.locA, sys.thrA, 0), pt(sys.locA, sys.thrA, 1))
    B = (pt(sys.locB, sys.thrB, 0), pt(sys.locB, sys.thrB, 1))
    return A, B

def corrected(sys, p_ld, target):
    """[7, eq. (25)]: LDs keep a common coin (bias p_ld), the DFC fuses on a
    private coin with bias q chosen so that Pf = target. Pf is linear in q."""
    def pf_pd(q, h):
        tot = 0.0
        for sd, qs in ((0, q), (1, 1 - q)):
            for sl, ps, op in ((0, p_ld, sys.locA), (1, 1 - p_ld, sys.locB)):
                for U in sys.U:
                    if sys.fuse(sd, U):
                        tot += qs * ps * sys.prod(op, U, h)
        return tot
    f0, f1 = pf_pd(0.0, 0), pf_pd(1.0, 0)
    q = (target - f0) / (f1 - f0)
    return (pf_pd(q, 0), pf_pd(q, 1)), q

PAPER = {  # values as printed in the letter / in [7]
    "2-LD": dict(C=(0.2009, 0.8261), Wstar=(0.2640, 0.7600), Wp=(0.2324, 0.7930),
                 Cstar=(0.2009, 0.7005), Cstar_eps=(0.1909, 0.7074), sigma=0.913),
    "3-LD": dict(C=(0.1708, 0.8448), Wstar=(0.2046, 0.8210), Wp=(0.1826, 0.8386),
                 Cstar=(0.1708, 0.7974), Cstar_eps=(0.1608, 0.7925), sigma=0.910),
}

def check(name, got, want, tol=6e-4):
    ok = all(abs(g - w) <= tol for g, w in zip(np.atleast_1d(got), np.atleast_1d(want)))
    fmt = lambda v: "(" + ", ".join(f"{x:.4f}" for x in np.atleast_1d(v)) + ")"
    print(f"  {name:<10} computed {fmt(got):<20} letter {fmt(want):<20} {'MATCH' if ok else '** MISMATCH **'}")
    return ok

if __name__ == "__main__":
    allok = True
    for nm, sys, part in (("2-LD", SYS2, {1}), ("3-LD", SYS3, {2})):
        P = PAPER[nm]; p = sys.p
        print(f"{nm}  (p = {p}, alpha = {sys.alpha})")
        A, B = exact_AB(sys)
        C = (p * A[0] + (1 - p) * B[0], p * A[1] + (1 - p) * B[1])
        Wstar = point(sys, sys.f_G(p, 0), sys.f_G(p, 1))
        Wp = point(sys, sys.f_part(part, p, 0), sys.f_part(part, p, 1))
        Cstar, _ = corrected(sys, p, sys.alpha)
        p_eps = (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])
        Cst_e, _ = corrected(sys, p_eps, sys.alpha - EPS)
        sigma = (B[1] - A[1]) / (B[0] - A[0])
        for k, v in (("C", C), ("Wstar", Wstar), ("Wp", Wp), ("Cstar", Cstar),
                     ("Cstar_eps", Cst_e), ("sigma", sigma)):
            allok &= check(k, v, P[k])
    print("\nALL MATCH" if allok else "\nSOME MISMATCH")
