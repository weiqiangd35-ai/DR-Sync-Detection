"""Recompute at C_eps, ARL=1e4, precise calibration. Stages A/B/C, JSON-checkpointed.
Usage: python3 pipeline.py 2LD|3LD A|B|C
"""
import json, os, sys as _s, numpy as np
from systems import SYS2, SYS3, kl
from fastsim import first_passage, mean_se, calibrate, renewal
ARL = 1e4; EPS = 0.01
nm, stage = _s.argv[1], _s.argv[2]
sys = {"2LD": SYS2, "3LD": SYS3}[nm]
keep = {"2LD": ["G", "P{LD2}", "F"], "3LD": ["G", "P{LD3}", "P{LD2,LD3}", "F"]}[nm]
part = {"2LD": "P{LD2}", "3LD": "P{LD3}"}[nm]
RNG = {"2LD": dict(bank=(6.6, 7.4), mix=(5.6, 6.6), u=(3.0, 4.2)),
       "3LD": dict(bank=(6.0, 6.8), mix=(4.4, 5.4), u=(1.2, 2.2))}[nm]
p = (sys.B[0] - (sys.alpha - EPS)) / (sys.B[0] - sys.A[0])
f0 = sys.f_sync(p); M = sys.modes(p); f01 = sys.f_sync(p, h=1); M1 = sys.modes(p, h=1)
keys = list(f0.keys())
L = {m: np.array([np.log(M[m][k] / f0[k]) if M[m][k] > 0 else -50.0 for k in keys]) for m in M}
P0 = np.array([f0[k] for k in keys]); PM = {m: np.array([M[m][k] for k in keys]) for m in M}
fuse = np.array([1.0 if sys.fuse(k[0], k[1]) else 0.0 for k in keys])
pf = lambda v: float((v * fuse).sum())
bank = [L[m] for m in M]; names = list(M)
Lmix = np.log(sum(PM[m] for m in M) / len(M) / P0)
def marg(v):
    d = {}
    for k, x in zip(keys, v): d[k[1]] = d.get(k[1], 0) + x
    return d
m0, mp = marg(P0), marg(PM[part])
Lu = np.array([np.log(mp[k[1]] / m0[k[1]]) for k in keys])
os.makedirs("results", exist_ok=True)
fn = f"results/final_{nm}.json"
out = json.load(open(fn)) if os.path.exists(fn) else dict(p_eps=p, nM=len(M))
save = lambda: json.dump(out, open(fn, "w"), indent=1, default=float)

if stage == "A":
    out["h_star"], _ = calibrate(bank, P0, ARL, *RNG["bank"], N=5000, seed=3)
    out["h_mix"], _ = calibrate([Lmix], P0, ARL, *RNG["mix"], N=5000, seed=3)
    out["h_u"], _ = calibrate([Lu], P0, ARL, *RNG["u"], N=5000, seed=3)
    out["h_a"] = float(np.log(2 * len(M) * ARL))
    for tag, ch, h in (("bank", bank, out["h_star"]), ("mix", [Lmix], out["h_mix"]), ("u", [Lu], out["h_u"])):
        T, _ = first_passage(ch, P0, [h], N=5000, seed=99, maxT=int(40 * ARL))
        out[f"arl_check_{tag}"] = mean_se(T[0])
    print(f"{nm}: h*={out['h_star']:.3f} h_mix={out['h_mix']:.3f} h_u={out['h_u']:.3f} "
          f"h_a={out['h_a']:.2f} | indep. ARL check: bank {out['arl_check_bank'][0]:.0f}"
          f"±{out['arl_check_bank'][1]:.0f}, mix {out['arl_check_mix'][0]:.0f}"
          f"±{out['arl_check_mix'][1]:.0f}, u {out['arl_check_u'][0]:.0f}±{out['arl_check_u'][1]:.0f}")
    save(); raise SystemExit

hs, hm, hu, ha = out["h_star"], out["h_mix"], out["h_u"], out["h_a"]
if stage == "B":
    rows = {}
    for m in keep:
        I0 = kl(M[m], f0); I1 = kl(M1[m], f01); lmax = float(L[m].max()); D = pf(PM[m]) - sys.alpha
        T, _ = first_passage(bank, PM[m], [hs], N=8000, seed=11)
        dmc, dse, _ = mean_se(T[0]); bs, ba = (hs + lmax) / I0, (ha + lmax) / I0
        rows[m] = dict(I0=I0, I1=I1, Delta=D, lmax=lmax, bnd=bs, bnd_a=ba, dmc=dmc, dse=dse,
                       inv=D*bs/EPS, inv_a=D*ba/EPS, inv_ex=D*dmc/EPS)
        print(f"  {m:<12} I0={I0:.4f} I1={I1:.4f} D={D:.4f} bnd={bs:.1f} bnd_a={ba:.1f} "
              f"MC={dmc:.1f}±{dse:.1f} 1/l*={D*bs/EPS:.1f} 1/l_a={D*ba/EPS:.1f}", flush=True)
    out["rows"] = rows
    hg = [3.0, 4.0, 5.0, 6.0, 7.0]
    T, _ = first_passage(bank, PM["G"], hg, N=6000, seed=12)
    out["dvh"] = dict(h=hg, mc=[mean_se(r)[0] for r in T],
                      bnd=[(h + float(L["G"].max())) / kl(M["G"], f0) for h in hg])
    lmix = float(Lmix.max()); mix = {}
    for m in keep:
        dr = float((PM[m] * Lmix).sum())
        T, _ = first_passage([Lmix], PM[m], [hm], N=6000, seed=13, maxT=400000)
        mix[m] = dict(drift=dr, dmc=mean_se(T[0])[0], bnd=(hm + lmix) / dr)
    out["mix"] = mix
    T, _ = first_passage([Lu], PM[part], [hu], N=6000, seed=14, maxT=400000)
    out["u_delay"] = mean_se(T[0])
    dG = float((PM[part] * L["G"]).sum())
    T1, _ = first_passage([L["G"]], PM[part], [hs], N=4000, seed=15, maxT=600000)
    out["remark1"] = dict(drift=dG, G_under_part=mean_se(T1[0]),
                          bound_if_pos=((hs + float(L["G"].max())) / dG) if dG > 0 else None)
    print(f"  dvh mc={[round(x) for x in out['dvh']['mc']]} bnd={[round(x) for x in out['dvh']['bnd']]}")
    print(f"  mix={ {m:(round(v['dmc']),round(v['bnd'])) for m,v in mix.items()} }")
    print(f"  U-only delay={out['u_delay'][0]:.0f} vs bank {rows[part]['dmc']:.0f} -> "
          f"{out['u_delay'][0]/rows[part]['dmc']:.2f}x")
    print(f"  Remark1: drift E_part[l_G]={dG:+.4f}, G alone under part: "
          f"{out['remark1']['G_under_part'][0]:.0f}, bound if +: {out['remark1']['bound_if_pos']}")
    save(); raise SystemExit

if stage == "C":
    rows = out["rows"]
    def cls(m): return m if m in ("G", "F") else f"P{m.count('LD')}"
    ex = []; cl = []
    for tm in M:
        _, am = first_passage(bank, PM[tm], [hs], N=3000, seed=17, want_argmax_at=hs)
        am = am[am >= 0]
        ex.append(np.mean([names[a] == tm for a in am])); cl.append(np.mean([cls(names[a]) == cls(tm) for a in am]))
    out["iso"] = dict(exact=float(np.mean(ex)), by_class=float(np.mean(cl)))
    T1, _ = first_passage([L["G"]], P0, [hs], N=2000, seed=16, maxT=1500000)
    out["remark1"]["G_under_sync"] = mean_se(T1[0])
    g = rows["G"]; lb = EPS/(g["Delta"]*g["bnd"]); lx = EPS/(g["Delta"]*g["dmc"])
    ren = {t: renewal(bank, P0, PM["G"], fuse, lam, hs, N=1000, T=10000, seed=5)
           for t, lam in (("bnd", lb), ("ex_half", lx/2), ("ex_double", 2*lx))}
    out["renewal"] = dict(lam_bnd=lb, lam_ex=lx, slots=int(1e7), **ren)
    print(f"{nm}: iso exact={out['iso']['exact']:.2f} class={out['iso']['by_class']:.2f} | "
          f"G alone sync ARL={out['remark1']['G_under_sync'][0]:.0f} | 1/l_bnd={1/lb:.0f} 1/l_ex={1/lx:.0f}")
    print("  renewal:", {k:(round(v[0],4),round(v[1],4)) for k,v in ren.items()}, "alpha", sys.alpha)
    save()
