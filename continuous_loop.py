"""Continuously operating monitor with verification/repair (minor-revision
round, Reviewer 1 point 2 and Reviewer 4 point 4).

Unlike fastsim.renewal (zero-start, monitor inactive while synchronized,
instantaneous repair), the bank runs at every slot of healthy operation, so
  * the chart state at a failure onset is whatever healthy operation left it,
  * spurious alarms occur during healthy operation,
  * every alarm, true or spurious, is followed by a verification/repair phase
    of rho slots during which the stopgap keyed to the alarming chart's mhat is
    in force and the monitor is paused; the system then re-synchronizes and the
    bank restarts from zero.
Per-slot false-alarm probabilities are accumulated analytically from the state
(C_eps while healthy, alpha + Delta_m while failed, Table II entry (true, mhat)
during repair; a healthy system under correction keyed to mhat has the law of
row G of Table II), so the long-run Pf has no binomial noise; the standard
error is across the N independent processes.

Usage: python3 continuous_loop.py <2LD|3LD> [rho] [N T]     (default rho = 100,
       N = 250 processes x T = 40000 slots = 1e7 slots per configuration)
Writes results/continuous_<sys>_rho<rho>.json.
"""
import json, sys as _s
import numpy as np
from systems import SYS2, SYS3
import mismatch_table as mt

EPS = 0.01
nm = _s.argv[1]
RHO = int(_s.argv[2]) if len(_s.argv) > 2 else 100
sys = {"2LD": SYS2, "3LD": SYS3}[nm]
res = json.load(open(f"results/final_{nm}.json"))
h = res["h_star"]
N, T = 250, 40000                      # 1e7 slots per configuration
if len(_s.argv) > 3: N, T = int(_s.argv[3]), int(_s.argv[4])   # override

p = (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])
f0 = sys.f_sync(p); M = sys.modes(p)
keys = list(f0.keys()); names = list(M.keys()); iG = names.index("G")
P0 = np.array([f0[k] for k in keys])
PM = np.array([[M[m][k] for k in keys] for m in names])            # (modes, K)
L = np.array([[np.log(M[m][k] / f0[k]) if M[m][k] > 0 else -50.0 for k in keys] for m in names])
fuse = np.array([1.0 if sys.fuse(k[0], k[1]) else 0.0 for k in keys])
pf_healthy = float(P0 @ fuse)                                       # alpha - eps
pf_failed = PM @ fuse                                               # alpha + Delta_m
mt.EPS = EPS
_, pf_tab, _ = mt.table(sys)                                        # Table II
pf_rep = np.array([[pf_tab[(names[i], names[j])] for j in range(len(names))] for i in range(len(names))])
cdf0, cdfM = np.cumsum(P0), np.cumsum(PM, axis=1)


def run(lam, mode_dist, seed):
    """mode_dist: probability vector over bank modes for the mode of each failure"""
    rng = np.random.default_rng(seed)
    Z = np.zeros((len(names), N))
    state = np.zeros(N, int)            # 0 healthy, 1 failed, 2 repair
    mode = np.zeros(N, int)             # mode in force (failed) / true row for repair
    mhat = np.zeros(N, int)
    rep_left = np.zeros(N, int)
    onset = np.full(N, -1)
    pf_acc = np.zeros(N)
    n_fail = np.zeros(N); n_true = np.zeros(N); n_spur = np.zeros(N); delay_sum = np.zeros(N)
    cdf_mode = np.cumsum(mode_dist)
    for t in range(T):
        # ---- accumulate the per-slot false-alarm probability of the current state
        pf_acc += np.where(state == 0, pf_healthy,
                  np.where(state == 1, pf_failed[mode], pf_rep[mode, mhat]))
        # ---- repair countdown
        inrep = state == 2
        rep_left[inrep] -= 1
        done = inrep & (rep_left <= 0)
        state[done] = 0; Z[:, done] = 0.0                               # re-synchronized, bank restarted
        # ---- monitor runs while healthy or failed
        act = state < 2
        if act.any():
            ai = np.flatnonzero(act)
            u = rng.random(ai.size)
            x = np.searchsorted(cdf0, u)                                   # healthy draw
            fsel = state[ai] == 1
            if fsel.any():
                for k in np.unique(mode[ai[fsel]]):                      # per-mode draw, vectorized
                    sel = fsel & (mode[ai] == k)
                    x[sel] = np.searchsorted(cdfM[k], u[sel])
            x = np.minimum(x, len(keys) - 1)
            Za = np.maximum(0.0, Z[:, ai] + L[:, x])
            Z[:, ai] = Za
            hit = Za.max(axis=0) >= h
            if hit.any():
                hi = ai[hit]
                mh = Za[:, hit].argmax(axis=0)
                was_failed = state[hi] == 1
                n_true += np.bincount(hi[was_failed], minlength=N) > 0
                n_spur += np.bincount(hi[~was_failed], minlength=N) > 0
                delay_sum[hi[was_failed]] += t - onset[hi[was_failed]]
                mode[hi[~was_failed]] = iG                                # healthy under correction = row G
                mhat[hi] = mh
                state[hi] = 2; rep_left[hi] = RHO
                if RHO == 0:                                              # instantaneous repair
                    state[hi] = 0; Z[:, hi] = 0.0
        # ---- failure onsets during healthy operation
        healthy = state == 0
        new = healthy & (rng.random(N) < lam)
        if new.any():
            ni = np.flatnonzero(new)
            mode[ni] = np.minimum(np.searchsorted(cdf_mode, rng.random(ni.size)), len(names) - 1)
            state[ni] = 1; onset[ni] = t; n_fail[ni] += 1
    per = pf_acc / T
    return dict(pf=float(per.mean()), se=float(per.std(ddof=1) / np.sqrt(N)),
                failures=float(n_fail.sum()), detected=float(n_true.sum()), spurious=float(n_spur.sum()),
                mean_delay=float(delay_sum.sum() / max(n_true.sum(), 1)),
                spurious_per_slot=float(n_spur.sum() / (N * T)))


if __name__ == "__main__":
    rows = res["rows"]; Dbnd = {m: rows[m]["bnd"] for m in rows}; Dmc = {m: rows[m]["dmc"] for m in rows}
    Delta = {m: rows[m]["Delta"] for m in rows}
    lam_bnd = EPS / (Delta["G"] * Dbnd["G"]); lam_ex = EPS / (Delta["G"] * Dmc["G"])
    onlyG = np.zeros(len(names)); onlyG[iG] = 1.0
    uniform = np.ones(len(names)) / len(names)
    # mixed-mode sufficient rate (min over modes of lambda*) and the exact mixed-mode bound rate
    full_bnd = {m: (res["rows"][m]["bnd"] if m in res["rows"] else None) for m in names}
    # map every bank mode to its symmetry-class representative in Table I
    def rep(m):                      # symmetry-class representative listed in Table I
        if m in rows: return m
        k = m.count("LD")
        cands = [r for r in rows if r.startswith("P{") and r.count("LD") == k]
        return cands[0]
    lam_min = min(EPS / (Delta[rep(m)] * Dbnd[rep(m)]) for m in names if Delta[rep(m)] > 0)
    lam_mix = EPS / np.mean([Delta[rep(m)] * Dbnd[rep(m)] for m in names])
    out = dict(rho=RHO, h=h, alpha=sys.alpha, slots=N * T, lam_bnd=lam_bnd, lam_ex=lam_ex,
               lam_min_over_modes=lam_min, lam_mixed_uniform=lam_mix, runs={})
    configs = [("G_at_lam_bnd", lam_bnd, onlyG), ("G_at_lam_ex", lam_ex, onlyG), ("G_at_2lam_ex", 2 * lam_ex, onlyG),
               ("mixed_at_lam_min", lam_min, uniform), ("mixed_at_lam_mix", lam_mix, uniform)]
    for i, (tag, lam, dist) in enumerate(configs):
        r = run(lam, dist, seed=100 + i)
        out["runs"][tag] = dict(lam=lam, inv_lam=1 / lam, **r)
        print(f"{nm} rho={RHO} {tag:<18} 1/lam={1/lam:6.0f}: Pf={r['pf']:.4f}+/-{r['se']:.5f} (alpha {sys.alpha}) "
              f"failures={r['failures']:.0f} detected={r['detected']:.0f} spurious={r['spurious']:.0f} "
              f"mean delay={r['mean_delay']:.0f} (zero-start {Dmc['G']:.0f}, bound {Dbnd['G']:.0f}) "
              f"spurious/slot={r['spurious_per_slot']:.2e}", flush=True)
    json.dump(out, open(f"results/continuous_{nm}_rho{RHO}.json", "w"), indent=1)
