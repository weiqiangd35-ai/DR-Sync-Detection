"""The two robust variants of Sec. VI, calibrated and measured as banks.

Each variant is an ordinary CUSUM bank on an i.i.d. per-slot record, so the
procedure of pipeline.py applies unchanged:
  * deployed bank : increments ln f_m(x|H0)/f_0(x|H0)                (reference)
  * mixture bank  : increments ln[(f_m(x|H0)+f_m(x|H1)) / (f_0(x|H0)+f_0(x|H1))]
  * gated bank    : charts run only on slots with u0 = 0, with both laws
                    conditioned on that event.  Implemented on the alphabet
                    {records with u0 = 0} + {skip}: a slot with u0 = 1 is the
                    symbol "skip" with increment 0 on every chart, so ARL and
                    delays come out directly in real slots.
For every bank and both systems, at C_eps:
  1. h calibrated to bank ARL = 1e4 under synchronization with H0 (same grid
     interpolation and independent-seed check as pipeline.py);
  2. spurious-alarm probability for an H1 episode of L slots (zero start, 100
     H0 slots appended), and the mean time to a spurious alarm under sustained H1;
  3. mode-G delay with H0 in force and with H1 in force (a failure coinciding
     with a target), zero start;
  4. drift of the G chart under mode G with H1 in force, the delay bound (4)
     and the critical rate (6) for mode G, bound-based and with the measured delay.
Finally, the weight interval of the mixture variant over which the G chart's
drift is negative under synchronization and positive under mode G for both
hypotheses.  Source of the variant rows of Table V and of the sentences in
Sec. VI that quote them.

Usage: python3 variant_calibration.py <2LD|3LD> <deployed|gated|mixture> <cal|check|meas>
         (1-3 min each; writes results/variants_<sys>_<bank>.json; run cal, then check, then meas)
       python3 variant_calibration.py report                                (prints the table)
"""
import json
import numpy as np
from systems import SYS2, SYS3, kl
from fastsim import first_passage, calibrate, mean_se

EPS = 0.01
ARL = 1e4
L_EPISODES = (10, 20, 30, 50, 80)
POST = 100
N_EP = 4000
N_DELAY = 8000
N_SUST = 500
CAP_SUST = 50000
CAP_DELAY = 20000


def p_eps(sys):
    return (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])


def laws(sys, p):
    f0, f1 = sys.f_sync(p, 0), sys.f_sync(p, 1)
    M0, M1 = sys.modes(p, 0), sys.modes(p, 1)
    keys = list(f0.keys())
    return f0, f1, M0, M1, keys


def vec(law, keys):
    return np.array([law.get(k, 0.0) for k in keys])


def llr(num, den, keys):
    return np.array([np.log(num[k] / den[k]) if num.get(k, 0) > 0 else -50.0 for k in keys])


def deployed_bank(f0, f1, M0, M1, keys):
    charts = [llr(M0[m], f0, keys) for m in M0]
    regimes = dict(sync_H0=vec(f0, keys), sync_H1=vec(f1, keys),
                   G_H0=vec(M0["G"], keys), G_H1=vec(M1["G"], keys))
    return charts, regimes, list(M0.keys())


def mixture_bank(f0, f1, M0, M1, keys, w=0.5):
    den = {k: (1 - w) * f0[k] + w * f1[k] for k in keys}
    charts = []
    for m in M0:
        num = {k: (1 - w) * M0[m][k] + w * M1[m][k] for k in keys}
        charts.append(llr(num, den, keys))
    regimes = dict(sync_H0=vec(f0, keys), sync_H1=vec(f1, keys),
                   G_H0=vec(M0["G"], keys), G_H1=vec(M1["G"], keys))
    return charts, regimes, list(M0.keys())


def gated_bank(sys, f0, f1, M0, M1, keys):
    kept = [k for k in keys if sys.fuse(k[0], k[1]) == 0]

    def cond(f):
        mass = sum(f[k] for k in kept)
        return {k: f[k] / mass for k in kept}

    f0c = cond(f0)
    charts = []
    for m in M0:
        charts.append(np.append(llr(cond(M0[m]), f0c, kept), 0.0))   # last symbol: skip
    regimes = {}
    for name, f in (("sync_H0", f0), ("sync_H1", f1), ("G_H0", M0["G"]), ("G_H1", M1["G"])):
        v = np.array([f[k] for k in kept])
        regimes[name] = np.append(v, 1.0 - v.sum())
    return charts, regimes, list(M0.keys())


def episode_prob(rng, charts, P1, P0, h, Lh, N):
    A = np.stack(charts)
    Z = np.zeros((A.shape[0], N))
    alarmed = np.zeros(N, bool)
    for _ in range(Lh):
        x = rng.choice(len(P1), size=N, p=P1)
        Z = np.maximum(0.0, Z + A[:, x])
        alarmed |= Z.max(0) >= h
    for _ in range(POST):
        x = rng.choice(len(P0), size=N, p=P0)
        Z = np.maximum(0.0, Z + A[:, x])
        alarmed |= Z.max(0) >= h
    return float(alarmed.mean())


def coarse_locate(charts, P0, hlo=3.0, hhi=9.0, step=0.5, N=1500, horizon=int(6 * ARL)):
    """where does ARL(h) cross the target? short horizon: thresholds far above the
    crossing are censored, which does not move the crossing itself"""
    hgrid = np.arange(hlo, hhi + 1e-9, step)
    T, _ = first_passage(charts, P0, hgrid, N=N, seed=3, maxT=horizon)
    arl = np.array([np.nanmean(np.where(np.isfinite(r), r, np.nan)) for r in T])
    j = int(np.searchsorted(arl, ARL)); j = min(max(j, 1), len(hgrid) - 1)
    h0, h1 = hgrid[j - 1], hgrid[j]; a0, a1 = np.log(arl[j - 1]), np.log(arl[j])
    return float(h0 + (np.log(ARL) - a0) * (h1 - h0) / (a1 - a0))


def fine_calibrate(charts, P0, hc):
    """the fine grid of pipeline.py around the coarse crossing"""
    h, _ = calibrate(charts, P0, ARL, hc - 0.3, hc + 0.25, step=0.025, N=5000, seed=3)
    return h


SYSTEMS = {"3LD": (SYS3, "final_3LD.json", 0.0216), "2LD": (SYS2, "final_2LD.json", 0.0496)}


def build(sys, p):
    f0, f1, M0, M1, keys = laws(sys, p)
    return {"deployed": deployed_bank(f0, f1, M0, M1, keys),
            "gated": gated_bank(sys, f0, f1, M0, M1, keys),
            "mixture": mixture_bank(f0, f1, M0, M1, keys)}, (f0, f1, M0, M1, keys)


def _load(fnv):
    try:
        return json.load(open(fnv))
    except FileNotFoundError:
        return {}


def run_one(nm, name, stage):
    sys, fn, Delta_G = SYSTEMS[nm]
    p = p_eps(sys)
    banks, _ = build(sys, p)
    charts, R, modes = banks[name]
    iG = modes.index("G")
    fnv = f"results/variants_{nm}_{name}.json"
    out = _load(fnv)
    if stage == "cal":
        hc = coarse_locate(charts, R["sync_H0"])
        out["h"] = fine_calibrate(charts, R["sync_H0"], hc)
        out["drift"] = {r: float(R[r] @ charts[iG]) for r in R}
        out["lmax"] = float(charts[iG].max())
        out["Dbnd"] = (out["h"] + out["lmax"]) / out["drift"]["G_H0"]
        out["inv_lambda_bound"] = Delta_G * out["Dbnd"] / EPS
        print(f"{nm} {name}: coarse {hc:.2f} -> h = {out['h']:.3f}; G-chart drifts {out['drift']}; "
              f"l_max {out['lmax']:.3f}; bound {out['Dbnd']:.0f}; 1/lambda* {out['inv_lambda_bound']:.0f}")
    elif stage == "check":
        T, _ = first_passage(charts, R["sync_H0"], [out["h"]], N=5000, seed=99, maxT=int(40 * ARL))
        chk, se, _ = mean_se(T[0]); out["arl_check"] = [chk, se]
        print(f"{nm} {name}: independent ARL check at h = {out['h']:.3f}: {chk:.0f} +/- {se:.0f}")
    elif stage == "meas":
        rng = np.random.default_rng(2026); h = out["h"]
        probs = [episode_prob(rng, charts, R["sync_H1"], R["sync_H0"], h, Lh, N_EP) for Lh in L_EPISODES]
        out["episodes"] = dict(zip(map(str, L_EPISODES), probs))
        Ts, _ = first_passage(charts, R["sync_H1"], [h], N=N_SUST, seed=21, maxT=CAP_SUST)
        fin = np.isfinite(Ts[0])
        out["sustained_H1"] = dict(mean_finite=float(Ts[0][fin].mean()) if fin.any() else None,
                                   frac_censored=float((~fin).mean()), cap=CAP_SUST)
        T0, _ = first_passage(charts, R["G_H0"], [h], N=N_DELAY, seed=11, maxT=CAP_DELAY)
        T1, _ = first_passage(charts, R["G_H1"], [h], N=N_DELAY, seed=12, maxT=CAP_DELAY)
        def summ(T):
            fin = np.isfinite(T[0])
            m, se, _ = mean_se(T[0][fin]) if fin.any() else (None, None, 0)
            return [m, se, float((~fin).mean())]          # mean of finite, s.e., censored fraction
        out["delay_G_H0"] = summ(T0); out["delay_G_H1"] = summ(T1)
        d0, d0se = out["delay_G_H0"][0], out["delay_G_H0"][1]; d1, d1se = out["delay_G_H1"][0], out["delay_G_H1"][1]
        out["inv_lambda_measured"] = Delta_G * d0 / EPS
        print(f"{nm} {name}: episodes {['%.3f' % x for x in probs]}; sustained-H1 {out['sustained_H1']}; "
              f"delay G+H0 {out['delay_G_H0']}, G+H1 {out['delay_G_H1']} (mean, s.e., censored at {CAP_DELAY}); "
              f"1/lambda*_ex {out['inv_lambda_measured']:.0f}")
    json.dump(out, open(fnv, "w"), indent=1)


def report():
    for nm, (sys, fn, Delta_G) in SYSTEMS.items():
        allres = {b: json.load(open(f"results/variants_{nm}_{b}.json")) for b in ("deployed", "gated", "mixture")}
        res = json.load(open(f"results/{fn}"))
        print(f"\n===== {nm}: Table I h* = {res['h_star']:.2f}, deployed mode-G delay {res['rows']['G']['dmc']:.0f} =====")
        print(f"{'bank':9s} {'h':>5s} {'ARLchk':>7s} | drift G: syncH0  syncH1   G+H0    G+H1 | "
              f"P(alarm|L=10/20/30/50/80) | sust.H1 | delay G+H0 G+H1 | bound 1/l* 1/l*_ex")
        for name in ("deployed", "gated", "mixture"):
            o = allres[name]; d = o["drift"]; s = o["sustained_H1"]
            sust = (f"{s['mean_finite']:.0f}" if s["frac_censored"] == 0 else
                    f">{s['cap']} ({s['frac_censored']:.0%} cens.)")
            dl = lambda v: (f"{v[0]:5.0f}" if v[2] == 0 else f"{v[0] if v[0] else float('nan'):5.0f}({v[2]:.0%}c)")
            print(f"{name:9s} {o['h']:5.2f} {o['arl_check'][0]:7.0f} | {d['sync_H0']:+.4f} {d['sync_H1']:+.4f} "
                  f"{d['G_H0']:+.4f} {d['G_H1']:+.4f} | "
                  + "/".join(f"{o['episodes'][k]:.3f}" for k in map(str, L_EPISODES))
                  + f" | {sust:>9s} | {dl(o['delay_G_H0'])} {dl(o['delay_G_H1'])} | "
                  f"{o['Dbnd']:5.0f} {o['inv_lambda_bound']:5.0f} {o['inv_lambda_measured']:5.0f}")
        p = p_eps(sys); banks, (f0, f1, M0, M1, keys) = build(sys, p)
        # analytic ARL bound of R1 needs E_{f0}[exp(l_m)] <= 1 for every chart
        for name, (charts, R, modes) in banks.items():
            mgf = [float(R["sync_H0"] @ np.exp(c)) for c in charts]
            print(f"E_sync-H0[exp(l_m)] over charts, {name:9s}: min {min(mgf):.4f}, max {max(mgf):.4f}"
                  f"{'  (<= 1: analytic ARL bound of R1 applies)' if max(mgf) <= 1 + 1e-9 else '  (> 1: MC calibration only)'}")
        ok = []
        for w in np.arange(0.01, 1.0, 0.01):
            ch, R, modes = mixture_bank(f0, f1, M0, M1, keys, w)
            g = ch[modes.index("G")]
            if (R["sync_H0"] @ g < 0) and (R["sync_H1"] @ g < 0) and (R["G_H0"] @ g > 0) and (R["G_H1"] @ g > 0):
                ok.append(w)
        print(f"mixture weight on H1 with sync drifts < 0 and mode-G drifts > 0 under both hypotheses: "
              + (f"[{min(ok):.2f}, {max(ok):.2f}]" if ok else "none"))


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) == 2 and _sys.argv[1] == "report":
        report()
    elif len(_sys.argv) == 4:
        run_one(_sys.argv[1], _sys.argv[2], _sys.argv[3])
    else:
        print(__doc__)
