"""
trajectory_plot.py -- plotting helpers for Fig. 3 (CUSUM trajectory).
========================================================
Produces, for each system (2-LD and 3-LD, Mode IV desynchronization):

  FIG 1 (main figure of the letter): one simulated trajectory, two stacked
        panels sharing the time axis.
        (a) CUSUM statistic W_t, threshold h, change point nu, alarm time T.
        (b) sliding-window empirical false-alarm rate of the global decision
            u0 (all slots simulated under H0, monitoring regime R1), with the
            theoretical piecewise levels alpha -> Pf(W*) -> Pf(C*) overlaid.
        Phases shaded: synchronized / undetected desync / corrected.

  FIG 2 (validation): Monte-Carlo mean detection delay vs ln(estimated ARL),
        overlaid with the asymptote  delay = ln(ARL)/I_m(H0)  used in the
        closed-loop critical-rate formula  lambda* = eps*I/(Delta*ln gamma).

Cross-checks printed to console:
  * post-correction operating point reproduces the parent paper's C*
    (2-LD: q=0.6787 -> C*=(0.2009,0.7005), Table V;
     3-LD: q=0.7033 -> C*=(0.1708,0.7974), Table VI).

Conventions:
  * Everything is simulated under H0 (regime R1), so panel (b) is a direct
    empirical Pf.  State this in the letter's caption.
  * The plotted run is chosen by a small seed search so that its detection
    delay is within [0.6, 1.5] x (h / I) -- i.e. a *representative* run, not a
    cherry-picked fast one.  Disclose in the caption ("a representative
    realization").

Outputs (PDF vector for LaTeX + PNG for quick viewing):
  fig_main_2LD.{pdf,png}   fig_main_3LD.{pdf,png}
  fig_delay_valid_2LD.{pdf,png}   fig_delay_valid_3LD.{pdf,png}
"""

import numpy as np
from itertools import product
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------
# IEEE letter styling (single column ~3.5 in)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.9,
    "axes.linewidth": 0.6,
    "figure.dpi": 200,
})

C_SYNC, C_DESY, C_CORR = "#f0f0f0", "#fbe3d5", "#ddeaf6"   # phase shading
C_STAT, C_PF = "#1f4e79", "#7a1f1f"


# ======================================================================
# System definitions (identical to exp1_v2.py, trimmed to what we need)
# ======================================================================
class System:
    def __init__(self, tag, n, locA, locB, thrA, thrB, p, alpha,
                 q_corr, Cstar_paper):
        self.tag, self.n = tag, n
        self.locA, self.locB = locA, locB
        self.thrA, self.thrB = thrA, thrB
        self.p, self.alpha = p, alpha
        self.q_corr = q_corr                 # eq.(25) correction prob (paper)
        self.Cstar_paper = Cstar_paper
        self.ALL_U = list(product([0, 1], repeat=n))

    @staticmethod
    def _bit(op, H, u):
        q = op[0] if H == 0 else op[1]
        return q if u == 1 else 1 - q

    def _prodU(self, op, H, U):
        pr = 1.0
        for u in U:
            pr *= self._bit(op, H, u)
        return pr

    # laws of (s,U) under H (s = DFC coin) --------------------------------
    def law_sync(self, H):
        law = {}
        for s, ps in (("A", self.p), ("B", 1 - self.p)):
            op = self.locA if s == "A" else self.locB
            for U in self.ALL_U:
                law[(s, U)] = ps * self._prodU(op, H, U)
        return law

    def law_modeIV(self, H):
        """LD group shares one coin s' independent of the DFC coin s."""
        g = {U: self.p * self._prodU(self.locA, H, U)
                + (1 - self.p) * self._prodU(self.locB, H, U)
             for U in self.ALL_U}
        law = {}
        for s, ps in (("A", self.p), ("B", 1 - self.p)):
            for U in self.ALL_U:
                law[(s, U)] = ps * g[U]
        return law

    def kl_H0(self):
        f0, fm = self.law_sync(0), self.law_modeIV(0)
        return sum(pm * np.log(pm / f0[k]) for k, pm in fm.items() if pm > 0)

    # LLR lookup table for the CUSUM (under H0) ---------------------------
    def llr_table(self):
        f0, fm = self.law_sync(0), self.law_modeIV(0)
        return {k: np.log(fm[k] / f0[k]) for k in f0}

    # increment distributions (values, probs) under each law --------------
    def increment_dist(self, under):
        tab = self.llr_table()
        law = self.law_sync(0) if under == "sync" else self.law_modeIV(0)
        keys = list(tab)
        vals = np.array([tab[k] for k in keys])
        prob = np.array([law[k] for k in keys])
        return vals, prob / prob.sum()

    # fusion output for (s, U)
    def fuse(self, s, U):
        thr = self.thrA if s == "A" else self.thrB
        return 1 if sum(U) >= thr else 0

    # theoretical Pf levels: sync C, desync W*, corrected C* ---------------
    def pf_levels(self):
        def pf_of(law):
            return sum(pr for (s, U), pr in law.items()
                       if self.fuse(s, U) == 1)
        pf_C = pf_of(self.law_sync(0))
        pf_W = pf_of(self.law_modeIV(0))
        # corrected: DFC coin bias q_corr, LD common coin bias p
        q = self.q_corr
        pf_Cs = pd_Cs = 0.0
        for s, ps in (("A", q), ("B", 1 - q)):
            for sp, psp in (("A", self.p), ("B", 1 - self.p)):
                op = self.locA if sp == "A" else self.locB
                for U in self.ALL_U:
                    w = ps * psp
                    if self.fuse(s, U) == 1:
                        pf_Cs += w * self._prodU(op, 0, U)
                        pd_Cs += w * self._prodU(op, 1, U)
        return pf_C, pf_W, (pf_Cs, pd_Cs)


SYS2 = System("2LD", 2, (0.3976, 0.8871), (0.1304, 0.6328), 2, 1,
              p=0.5, alpha=0.2009, q_corr=0.6787,
              Cstar_paper=(0.2009, 0.7005))
SYS3 = System("3LD", 3, (0.2, 0.7), (0.1, 0.6), 2, 1,
              p=0.6, alpha=0.1708, q_corr=0.7033,
              Cstar_paper=(0.1708, 0.7974))


# ======================================================================
# Single-run trajectory simulation (all under H0)
# ======================================================================
def simulate_run(sys, nu, T_total, h, seed):
    """Returns dict with W (CUSUM), u0 (global decisions), alarm time."""
    rng = np.random.default_rng(seed)
    tab = sys.llr_table()
    W = np.zeros(T_total + 1)
    u0 = np.zeros(T_total, dtype=int)
    alarm = None
    for t in range(T_total):
        s = "A" if rng.random() < sys.p else "B"        # DFC coin
        if t < nu:                                       # synchronized
            opc = s
        else:                                            # Mode IV desync
            opc = "A" if rng.random() < sys.p else "B"   # LD-group coin s'
        # after the alarm, the DFC applies the corrective bias q for ITS rule
        if alarm is not None:
            s = "A" if rng.random() < sys.q_corr else "B"
        op = sys.locA if opc == "A" else sys.locB
        U = tuple(int(rng.random() < op[0]) for _ in range(sys.n))  # H0 bits
        u0[t] = sys.fuse(s, U)
        if alarm is None:
            W[t + 1] = max(0.0, W[t] + tab[(s, U)])
            if W[t + 1] >= h and t >= 1:
                alarm = t + 1
        else:
            W[t + 1] = 0.0                               # reset & freeze
    return {"W": W, "u0": u0, "alarm": alarm}


def representative_seed(sys, nu, T_total, h, I, max_tries=500, tol=0.015):
    """First seed that is typical in BOTH displayed dimensions:
    (i) delay in [0.6, 1.5] x (h/I), alarm > nu;
    (ii) pre-change and post-correction empirical Pf within +-tol of alpha
    (both the sync point C and the corrected point C* sit exactly at alpha,
    so a typical trace straddles the alpha line in both phases)."""
    target = h / I
    for seed in range(max_tries):
        r = simulate_run(sys, nu, T_total, h, seed)
        if not (r["alarm"] and r["alarm"] > nu):
            continue
        d = r["alarm"] - nu
        if not (0.6 * target <= d <= 1.5 * target):
            continue
        pre = r["u0"][:nu].mean()
        if abs(pre - sys.alpha) > tol:
            continue
        if T_total - r["alarm"] < 200:
            continue
        post = r["u0"][r["alarm"]:].mean()
        if abs(post - sys.alpha) > tol:
            continue
        return seed, r
    raise RuntimeError("no representative seed found; widen the bands")


# ======================================================================
# FIG 1 -- main letter figure
# ======================================================================
def fig_main(sys, nu, T_total, h, win, fname):
    """Single-panel realization: the bank's CUSUM statistic, flat under
    synchrony, climbing after the change at nu, crossing h at the alarm.
    (The sliding-Pf panel was dropped: at the MC-calibrated threshold the
    detection delay is far shorter than any window long enough to suppress
    binomial noise, so that panel showed noise rather than the transient.)"""
    I = sys.kl_H0()
    seed, r = representative_seed(sys, nu, T_total, h, I)
    alarm = r["alarm"]
    fig, ax = plt.subplots(figsize=(3.5, 1.15))
    ax.axvspan(0, nu, color=C_SYNC, lw=0)
    ax.axvspan(nu, alarm, color=C_DESY, lw=0)
    ax.plot(np.arange(alarm + 1), r["W"][:alarm + 1], color=C_STAT, lw=0.9)
    ax.axhline(h, color=C_STAT, ls="--", lw=0.7)
    ax.text(0.015 * alarm, h * 1.02, "$h$", color=C_STAT, va="bottom",
            fontsize=7)
    ax.axvline(nu, color="0.35", ls=":", lw=0.8)
    ax.axvline(alarm, color="0.15", ls="--", lw=0.8)
    ax.text(nu - 0.035 * alarm, h * 1.19, r"$\nu$", ha="right", va="center", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(alarm - 0.035 * alarm, h * 1.19, "alarm", ha="right", va="center", fontsize=7,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(nu / 2, h * 1.19, "synchronized", ha="center", fontsize=7,
            color="0.35")
    ax.set_xlabel("time slot $t$")
    ax.set_ylabel(r"CUSUM $Z_t$")
    ax.set_xlim(0, alarm * 1.03)
    ax.set_ylim(0, h * 1.40)
    for ext in ("pdf", "png"):
        fig.savefig(f"{fname}.{ext}", bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"[{fname}] seed={seed} delay={alarm-nu}")
    return seed, alarm


def cusum_stop_times(vals, prob, h, n_runs, t_cap, rng):
    """Vectorized CUSUM first-passage times for iid increments (vals, prob)."""
    W = np.zeros(n_runs)
    stop = np.full(n_runs, t_cap, dtype=np.int64)
    active = np.ones(n_runs, dtype=bool)
    cdf = np.cumsum(prob)
    t = 0
    BLK = 4096
    buf = None; bpos = BLK
    while active.any() and t < t_cap:
        if bpos >= BLK:                                   # refill block
            u = rng.random((BLK, n_runs))
            buf = vals[np.searchsorted(cdf, u)]
            bpos = 0
        inc = buf[bpos]; bpos += 1; t += 1
        W[active] = np.maximum(0.0, W[active] + inc[active])
        hit = active & (W >= h)
        stop[hit] = t
        active &= ~hit
    return stop


