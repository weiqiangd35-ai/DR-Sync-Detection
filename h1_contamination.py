"""Sensitivity of the H0-designed monitor to sustained targets, two DFC-local
robust variants, and clock slip.  Source of every number in Sec. VI
("Sustained targets ...") and in R3 of the letter.

All quantities at the deployed design C_eps (eps = 0.01) and the
MC-calibrated bank threshold h* read from results/final_<sys>.json.

Parts
  A. expected increment of every chart under synchronization with H0 and with
     H1 in force (the latter is the "contamination drift"), and the i.i.d.
     target fraction at which the G chart's synchronized drift turns positive
  B. bank spurious-alarm time under a sustained target, and the probability
     of a spurious alarm raised by a target episode of L slots
  C. variant 1: charts gated on slots with u0 = 0 (laws conditional on u0 = 0)
  D. variant 2: equal-weight mixture over H0/H1 in both numerator and
     denominator of the G-chart increment
  E. clock slip: LDs apply s_{t-d}; per-slot law equals f_G, increments
     uncorrelated; measured G-chart delay vs. the i.i.d. mode-G delay

Usage: python3 h1_contamination.py        (about 1 min)
"""
import json
import numpy as np
from systems import SYS2, SYS3, kl

EPS = 0.01
SEED = 2026
L_EPISODES = (10, 20, 30, 50, 80)
POST_SLOTS = 100          # H0 slots appended after an episode before scoring
N_PATHS = 4000
N_SLIP = 6000


def p_eps(sys):
    return (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])


def increments(fm, f0, keys):
    return np.array([np.log(fm[k] / f0[k]) if fm[k] > 0 else -50.0 for k in keys])


def expect(law, keys, vec):
    return float(sum(law[k] * v for k, v in zip(keys, vec)))


def cusum_first_passage(rng, L, P, h, N, maxT):
    """bank of charts with increment table L (charts x alphabet), records i.i.d. ~ P"""
    Z = np.zeros((L.shape[0], N))
    T = np.full(N, maxT)
    alive = np.ones(N, bool)
    for t in range(1, maxT + 1):
        x = rng.choice(len(P), size=N, p=P)
        Z = np.maximum(0.0, Z + L[:, x])
        hit = alive & (Z.max(0) >= h)
        T[hit] = t
        alive &= ~hit
        if not alive.any():
            break
    return T


def episode_alarm_prob(rng, L, P1, P0, h, Lh, N):
    Z = np.zeros((L.shape[0], N))
    alarmed = np.zeros(N, bool)
    for _ in range(Lh):
        x = rng.choice(len(P1), size=N, p=P1)
        Z = np.maximum(0.0, Z + L[:, x])
        alarmed |= Z.max(0) >= h
    for _ in range(POST_SLOTS):
        x = rng.choice(len(P0), size=N, p=P0)
        Z = np.maximum(0.0, Z + L[:, x])
        alarmed |= Z.max(0) >= h
    return float(alarmed.mean())


def conditional(f, sys, u0=0):
    g = {k: v for k, v in f.items() if sys.fuse(k[0], k[1]) == u0}
    mass = sum(g.values())
    return {k: v / mass for k, v in g.items()}, mass


def slip_delay(rng, sys, p, f0, lG_tab, h, d, N):
    """G-chart delay (zero start) when every LD applies s_{t-d}; cs=0 is rule A"""
    keys = list(f0.keys())
    ukeys = sorted(set(k[1] for k in keys))
    uidx = {u: i for i, u in enumerate(ukeys)}
    Pu_s = np.zeros((2, len(ukeys)))
    for (cs, u), v in f0.items():
        Pu_s[cs, uidx[u]] = v
    Pu_s /= Pu_s.sum(1, keepdims=True)
    cdf = np.cumsum(Pu_s, 1)
    lut = np.zeros((2, len(ukeys)))
    for (cs, u) in keys:
        lut[cs, uidx[u]] = lG_tab[keys.index((cs, u))]
    Z = np.zeros(N)
    T = np.full(N, 50000)
    alive = np.ones(N, bool)
    hist = [(rng.random(N) >= p).astype(int) for _ in range(d)]
    for t in range(1, 50000):
        s_now = (rng.random(N) >= p).astype(int)        # Pr(cs = 0) = p
        s_used = hist[-d]
        hist.append(s_now)
        r = rng.random(N)
        u = np.where(s_used == 0, np.searchsorted(cdf[0], r), np.searchsorted(cdf[1], r))
        Z = np.maximum(0.0, Z + lut[s_now, u])
        hit = alive & (Z >= h)
        T[hit] = t
        alive &= ~hit
        if not alive.any():
            break
    return T


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    for nm, sys, fn in (("3-LD", SYS3, "final_3LD.json"), ("2-LD", SYS2, "final_2LD.json")):
        res = json.load(open(f"results/{fn}"))
        h = res["h_star"]
        p = p_eps(sys)
        f0 = sys.f_sync(p, 0)
        f1 = sys.f_sync(p, 1)
        M = sys.modes(p, 0)
        keys = list(f0.keys())
        P0 = np.array([f0[k] for k in keys])
        P1 = np.array([f1[k] for k in keys])
        L = np.array([increments(M[m], f0, keys) for m in M])
        names = list(M.keys())
        print(f"\n===== {nm}: C_eps (p_eps = {p:.4f}), h* = {h:.2f} =====")

        # ---- A. drifts under synchronization, H0 and H1 in force
        print("A. chart        I_m(H0)   E[l|sync,H0]  E[l|sync,H1]  ratio_H1/I  pi_crit")
        for i, m in enumerate(names):
            I = kl(M[m], f0)
            e0 = expect(f0, keys, L[i])
            e1 = expect(f1, keys, L[i])
            pic = -e0 / (e1 - e0) if e1 > e0 else float("nan")
            print(f"   {m:12s} {I:8.4f}   {e0:+10.4f}   {e1:+10.4f}   {e1 / I:8.2f}   {pic:6.3f}")
        iG = names.index("G")

        # ---- B. spurious alarms from sustained / episodic targets
        T = cusum_first_passage(rng, L, P1, h, N_PATHS, 5000)
        print(f"B. bank spurious alarm under sustained H1: mean {T.mean():.0f}, "
              f"median {np.median(T):.0f} slots ({N_PATHS} paths)")
        for Lh in L_EPISODES:
            pr = episode_alarm_prob(rng, L, P1, P0, h, Lh, N_PATHS)
            print(f"   P(spurious alarm | H1 episode of {Lh:3d} slots) = {pr:.3f}")

        # ---- C. variant 1: gate on u0 = 0
        f0c, kept0 = conditional(f0, sys)
        f1c, kept1 = conditional(f1, sys)
        gc, keptG = conditional(M["G"], sys)
        ck = list(f0c.keys())
        lg = increments(gc, f0c, ck)
        IGc = kl(gc, f0c)
        print(f"C. gated G chart: I_G = {IGc:.4f} per kept slot, kept fraction {kept0:.2f} under sync-H0 "
              f"({keptG:.2f} under G) -> {IGc * keptG:.4f} per slot, "
              f"{kl(M['G'], f0) / (IGc * keptG):.1f}x slower to first order; "
              f"drift under sync-H1 {expect(f1c, ck, lg):+.4f} per kept slot "
              f"(kept fraction {kept1:.2f}); drift under sync-H0 {expect(f0c, ck, lg):+.4f}")

        # ---- D. variant 2: hypothesis mixture in the G-chart increment (weight w on H1)
        fG1 = sys.f_G(p, 1)
        for w in (0.5, 0.05):
            mixG = {k: (1 - w) * M["G"][k] + w * fG1[k] for k in keys}
            mix0 = {k: (1 - w) * f0[k] + w * f1[k] for k in keys}
            lm = increments(mixG, mix0, keys)
            dG0, dG1 = expect(M["G"], keys, lm), expect(fG1, keys, lm)
            print(f"D. mixture G chart (w = {w}): drift under sync  H0 {expect(f0, keys, lm):+.4f}, "
                  f"H1 {expect(f1, keys, lm):+.4f}; under mode G  H0 {dG0:+.4f}, H1 {dG1:+.4f}; "
                  f"{kl(M['G'], f0) / dG0:.1f}x slower to first order"
                  + ("  <- deployed variant" if w == 0.5 else "  <- sign of the sync-H1 drift is design-dependent"))

        # ---- E. clock slip: per-slot law and increment autocorrelation on one long path
        ukeys = sorted(set(k[1] for k in keys)); uidx = {u: i for i, u in enumerate(ukeys)}
        Pu_s = np.zeros((2, len(ukeys)))
        for (cs, u), v in f0.items():
            Pu_s[cs, uidx[u]] = v
        Pu_s /= Pu_s.sum(1, keepdims=True); cdf = np.cumsum(Pu_s, 1)
        TT = 300000
        s_seq = (rng.random(TT) >= p).astype(int); r = rng.random(TT)
        u_seq = np.where(s_seq[:-1] == 0, np.searchsorted(cdf[0], r[1:]), np.searchsorted(cdf[1], r[1:]))
        lut = np.zeros((2, len(ukeys)))
        for (cs, u) in keys:
            lut[cs, uidx[u]] = L[iG][keys.index((cs, u))]
        inc = lut[s_seq[1:], u_seq]
        emp = np.zeros(len(keys))
        for (cs, u), i in zip(keys, range(len(keys))):
            emp[i] = np.mean((s_seq[1:] == cs) & (u_seq == uidx[u]))
        fG_vec = np.array([M["G"][k] for k in keys])
        print(f"E. clock slip d = 1, one path of {TT} slots: max |empirical law - f_G| = {np.abs(emp - fG_vec).max():.4f}, "
              f"mean increment {inc.mean():.4f} (I_G = {kl(M['G'], f0):.4f}), "
              f"lag-1 autocorrelation {np.corrcoef(inc[:-1], inc[1:])[0, 1]:+.3f}")
        for d in (1, 3):
            Td = slip_delay(rng, sys, p, f0, L[iG], h, d, N_SLIP)
            print(f"E. clock slip d = {d}: G-chart delay {Td.mean():.0f} +/- {Td.std() / np.sqrt(N_SLIP):.1f} "
                  f"(i.i.d. mode G: {res['D_MC']['G'] if 'D_MC' in res else 'Table I'}; bound Table I)")
